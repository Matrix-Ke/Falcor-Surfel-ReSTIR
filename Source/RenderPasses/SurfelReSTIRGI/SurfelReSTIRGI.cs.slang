#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Scene.RaytracingInline;
import Rendering.Lights.LightHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Reservoir;

static const uint kSlotsPerBucket = 4;
static const uint kTileSize = 8;
static const uint kEmpty = 0xffffffffu;
static const uint kEvictionAge = 120;
static const uint kMaxIrradianceSamples = 128;

struct Surfel
{
    float3 position;
    uint occupied;
    float3 normal;
    uint lastRequested;
    float3 geometricNormal;
};

struct Irradiance
{
    float3 value;
    uint samples;
};

cbuffer PerFrame
{
    uint2 gFrameDim;
    uint2 gTileDim;
    uint gFrame;
    uint gBucketCount;
    uint gRequestCapacity;
    uint gUpdateBudget;
    uint gUpdateStart;
    uint gUseCache;
    uint gUpdateCache;
    uint gAllowSurfelEviction;
    uint gUseTemporal;
    uint gUseSpatial;
    uint gHistoryValid;
    uint gCollectStats;
    uint gWriteIndirect;
    uint gWriteColor;
    uint gWriteSurfelOnly;
    uint gWriteSurfelDiscs;
    uint gWriteSurfaceAlbedo;
    uint gWriteSurfaceNormal;
    uint gWriteSurfaceDepth;
    uint gWriteCandidateIndirect;
    uint gWriteCandidateStatus;
    uint gWriteTemporalIndirect;
    uint gWriteHistorySamples;
    uint gUseAnalyticLights;
    float gCellSize;
    float3 gGridOrigin;
};

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;
Texture2D<float2> gMotion;
Texture2D<float4> gDirectIn;
RWTexture2D<float4> gDirect;
RWTexture2D<float4> gIndirect;
RWTexture2D<float4> gColor;
RWTexture2D<float4> gSurfelOnly;
RWTexture2D<float4> gSurfelDiscs;
RWTexture2D<float4> gSurfaceAlbedo;
RWTexture2D<float4> gSurfaceNormal;
RWTexture2D<float4> gSurfaceDepth;
RWTexture2D<float4> gCandidateIndirect;
RWTexture2D<float4> gCandidateStatus;
RWTexture2D<float4> gTemporalIndirect;
RWTexture2D<float4> gHistorySamples;

StructuredBuffer<Surfel> gSurfels;
RWStructuredBuffer<Surfel> gSurfelOut;
StructuredBuffer<Irradiance> gCache;
RWStructuredBuffer<Irradiance> gCacheOut;
RWBuffer<uint> gProposals;
RWBuffer<uint> gRequestCount;
StructuredBuffer<Surface> gRequests;
RWStructuredBuffer<Surface> gRequestOut;
RWBuffer<uint> gStats;

StructuredBuffer<Surface> gPrimary;
StructuredBuffer<Surface> gPreviousPrimary;
RWStructuredBuffer<Surface> gPrimaryOut;
StructuredBuffer<GIReservoir> gInput;
StructuredBuffer<GIReservoir> gHistory;
RWStructuredBuffer<GIReservoir> gReservoirOut;

enum Stat
{
    CandidateRay,
    ReconnectionRay,
    CacheRay,
    Lookup,
    CacheHit,
    DirectShadowRay,
    OccludedReconnection,
    RejectedReprojection,
};

void count(Stat stat)
{
    if (gCollectStats != 0)
        InterlockedAdd(gStats[uint(stat)], 1);
}

float3 sampleDirection(float3 normal, inout SampleGenerator rng, out float pdf)
{
    float3 tangent, bitangent;
    branchlessONB(normal, tangent, bitangent);
    float3 local = sample_cosine_hemisphere_concentric(sampleNext2D(rng), pdf);
    return local.x * tangent + local.y * bitangent + local.z * normal;
}

Surface loadSurface(HitInfo hit, float3 view)
{
    Surface surface = {};
    if (!hit.isValid() || hit.getType() != HitType::Triangle)
        return surface;

    TriangleHit triangle = hit.getTriangleHit();
    VertexData vertex = gScene.getVertexData(triangle);
    uint materialID = gScene.getMaterialID(triangle.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(vertex, materialID, view);
    if (!sd.frontFacing)
        return surface;

    let material = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0.f));
    surface.position = sd.posW;
    surface.instance = triangle.instanceID.index;
    surface.valid = 1;
    surface.geometricNormal = sd.faceN;
    surface.normal = sd.frame.N;
    surface.albedo = saturate(material.getProperties(sd).diffuseReflectionAlbedo);
    return surface;
}

Surface loadPrimary(uint2 pixel)
{
    return loadSurface(HitInfo(gVBuffer[pixel]), gViewW[pixel].xyz);
}

Surface traceSurface(Ray ray)
{
    SceneRayQuery<0> query;
    float hitT;
    HitInfo hit = query.traceRay(ray, hitT);
    return loadSurface(hit, -ray.dir);
}

float3 directLighting(Surface surface, inout SampleGenerator rng)
{
    uint lightCount = gScene.getLightCount();
    if (gUseAnalyticLights == 0 || lightCount == 0)
        return 0.f;

    uint index = min(uint(sampleNext1D(rng) * lightCount), lightCount - 1);
    AnalyticLightSample light;
    if (!sampleLight(surface.position, gScene.getLight(index), rng, light))
        return 0.f;

    float cosine = dot(surface.normal, light.dir);
    if (cosine <= 0.f || dot(surface.geometricNormal, light.dir) <= 0.f)
        return 0.f;

    float3 origin = computeRayOrigin(surface.position, surface.geometricNormal);
    float distance = light.distance < FLT_MAX ? dot(light.posW - origin, light.dir) : FLT_MAX;
    if (distance <= 0.f)
        return 0.f;

    SceneRayQuery<0> query;
    count(Stat::DirectShadowRay);
    bool visible = query.traceVisibilityRay(Ray(origin, light.dir, 0.f, distance));
    return visible ? light.Li * surface.albedo * (cosine * lightCount * M_1_PI) : float3(0.f);
}

int3 cellOf(float3 position)
{
    return int3(floor((position - gGridOrigin) / gCellSize));
}

uint bucketOf(int3 cell)
{
    uint3 bits = asuint(cell);
    uint hash = (bits.x * 73856093u) ^ (bits.y * 19349663u) ^ (bits.z * 83492791u);
    return hash & (gBucketCount - 1);
}

float surfelDistance(Surfel surfel, Surface surface)
{
    float3 offset = surface.position - surfel.position;
    if (dot(surfel.geometricNormal, surface.geometricNormal) < 0.9f ||
        dot(surfel.normal, surface.normal) < 0.8f ||
        abs(dot(offset, surfel.geometricNormal)) > gCellSize * 0.1f ||
        abs(dot(offset, surface.geometricNormal)) > gCellSize * 0.1f)
        return FLT_MAX;

    return length(offset) / gCellSize;
}

bool lookupSurfels(Surface surface, bool showDiscs, out float3 irradiance, out float3 discColor)
{
    const float discRadius = 0.33f;
    float closestDisc = discRadius;
    irradiance = 0.f;
    discColor = 0.03f * surface.albedo;
    float weightSum = 0.f;
    int3 center = cellOf(surface.position);
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        int3 cell = center + int3(x, y, z);
        uint offset = bucketOf(cell) * kSlotsPerBucket;
        for (uint i = 0; i < kSlotsPerBucket; ++i)
        {
            uint slot = offset + i;
            Surfel surfel = gSurfels[slot];
            if (surfel.occupied == 0 || any(cellOf(surfel.position) != cell))
                continue;

            float distance = surfelDistance(surfel, surface);
            if (distance >= 1.f)
                continue;

            Irradiance cached = gCache[slot];
            float confidence = saturate(float(cached.samples) / 16.f);
            float kernel = 1.f - distance * distance;
            float weight = kernel * kernel * confidence;
            irradiance += weight * cached.value;
            weightSum += weight;

            if (showDiscs && distance < closestDisc)
            {
                closestDisc = distance;
                float3 color = lerp(float3(0.95f, 0.27f, 0.06f), float3(0.1f, 0.83f, 0.58f), confidence);
                float edge = 1.f - smoothstep(discRadius - 0.05f, discRadius, distance);
                discColor = lerp(0.03f * surface.albedo, color, edge);
            }
        }
    }

    // Keep the fade at the edge of coverage, even when a pixel sees only one surfel.
    irradiance /= max(weightSum, 0.15f);
    return weightSum > 0.f;
}

bool lookupCache(Surface surface, out float3 irradiance)
{
    irradiance = 0.f;
    if (gUseCache == 0)
        return false;

    count(Stat::Lookup);
    float3 unused;
    if (!lookupSurfels(surface, false, irradiance, unused))
        return false;

    count(Stat::CacheHit);
    return true;
}

uint2 tilePixel(uint tileIndex)
{
    uint2 tile = uint2(tileIndex % gTileDim.x, tileIndex / gTileDim.x);
    uint offset = (gFrame + tileIndex * 13u) % (kTileSize * kTileSize);
    return min(tile * kTileSize + uint2(offset % kTileSize, offset / kTileSize), gFrameDim - 1);
}

Surface loadProposal(uint index, uint tileCount)
{
    if (index < tileCount)
        return loadPrimary(tilePixel(index));
    return gRequests[index - tileCount];
}

[numthreads(64, 1, 1)]
void resetCache(uint slot : SV_DispatchThreadID)
{
    if (slot < gBucketCount * kSlotsPerBucket)
    {
        gSurfelOut[slot] = {};
        gCacheOut[slot] = {};
    }
}

[numthreads(64, 1, 1)]
void proposeSurfels(uint index : SV_DispatchThreadID)
{
    uint tileCount = gTileDim.x * gTileDim.y;
    if (index >= tileCount + min(gRequestCount[0], gRequestCapacity))
        return;

    bool primary = index < tileCount;
    Surface surface = loadProposal(index, tileCount);
    if (surface.valid != 0)
    {
        // Alternate which source has priority when both want the same bucket.
        uint priority = primary == ((gFrame & 1) == 0) ? 0 : 0x80000000u;
        InterlockedMin(gProposals[bucketOf(cellOf(surface.position))], priority | index);
    }
}

[numthreads(64, 1, 1)]
void updateIrradiance(uint slot : SV_DispatchThreadID)
{
    uint capacity = gBucketCount * kSlotsPerBucket;
    if (slot >= capacity)
        return;

    Irradiance updated = gCache[slot];
    Surfel surfel = gSurfels[slot];
    uint relativeIndex = (slot + capacity - gUpdateStart) % capacity;
    if (surfel.occupied != 0 && relativeIndex < gUpdateBudget)
    {
        SampleGenerator rng = SampleGenerator(uint2(slot, 0x5a5au), gFrame);
        float pdf;
        float3 direction = sampleDirection(surfel.normal, rng, pdf);
        float3 incident = 0.f;
        if (dot(surfel.geometricNormal, direction) > 0.f)
        {
            count(Stat::CacheRay);
            Surface hit = traceSurface(Ray(computeRayOrigin(surfel.position, surfel.geometricNormal), direction));
            if (hit.valid != 0)
            {
                GISample bounce = {};
                bounce.surface = hit;
                bounce.direct = directLighting(hit, rng);
                float3 irradiance;
                lookupCache(hit, irradiance);
                bounce.setIrradiance(irradiance);
                incident = bounce.radiance();
            }
        }
        updated.samples = min(updated.samples + 1, kMaxIrradianceSamples);
        updated.value = lerp(updated.value, M_PI * incident, 1.f / updated.samples);
    }
    gCacheOut[slot] = updated;
}

[numthreads(64, 1, 1)]
void allocateSurfels(uint bucket : SV_DispatchThreadID)
{
    if (bucket >= gBucketCount || gProposals[bucket] == kEmpty)
        return;

    uint index = gProposals[bucket] & 0x7fffffffu;
    uint tileCount = gTileDim.x * gTileDim.y;
    Surface surface = loadProposal(index, tileCount);
    int3 cell = cellOf(surface.position);
    uint slot = kEmpty;
    uint oldestAge = kEvictionAge - 1;
    for (uint i = 0; i < kSlotsPerBucket; ++i)
    {
        uint currentSlot = bucket * kSlotsPerBucket + i;
        Surfel current = gSurfelOut[currentSlot];
        if (current.occupied == 0)
        {
            slot = currentSlot;
            oldestAge = kEmpty;
        }
        else if (all(cellOf(current.position) == cell) && surfelDistance(current, surface) < 0.5f)
        {
            current.lastRequested = gFrame;
            gSurfelOut[currentSlot] = current;
            return;
        }
        else if (gAllowSurfelEviction != 0 && gFrame - current.lastRequested > oldestAge)
        {
            slot = currentSlot;
            oldestAge = gFrame - current.lastRequested;
        }
    }

    if (slot != kEmpty)
    {
        Surfel surfel = {surface.position, 1, surface.normal, gFrame, surface.geometricNormal};
        gSurfelOut[slot] = surfel;
        gCacheOut[slot] = {};
    }
}

bool visibleConnection(Surface receiver, Surface endpoint)
{
    float3 origin = computeRayOrigin(receiver.position, receiver.geometricNormal);
    float3 end = computeRayOrigin(endpoint.position, endpoint.geometricNormal);
    float3 offset = end - origin;
    float distance = length(offset);
    if (distance <= 1e-6f)
        return false;

    SceneRayQuery<0> query;
    count(Stat::ReconnectionRay);
    bool visible = query.traceVisibilityRay(Ray(origin, offset / distance, 0.f, distance));
    if (!visible)
        count(Stat::OccludedReconnection);
    return visible;
}

bool compatible(Surface current, Surface other, float maxDistance, float maxPlaneDistance)
{
    float3 offset = other.position - current.position;
    return other.valid != 0 && current.instance == other.instance &&
        dot(current.normal, other.normal) >= 0.9f &&
        dot(current.geometricNormal, other.geometricNormal) >= 0.95f &&
        abs(dot(offset, current.geometricNormal)) <= maxPlaneDistance &&
        dot(offset, offset) <= maxDistance * maxDistance;
}

float worldPixelScale(Surface surface)
{
    return length(surface.position - gScene.camera.getPosition()) / float(min(gFrameDim.x, gFrameDim.y));
}

void merge(inout GIReservoir result, Surface receiver, GIReservoir source, inout SampleGenerator rng)
{
    float target = 0.f;
    if (source.W > 0.f)
    {
        float geometry;
        float value = luminance(contribution(receiver, source.sample, geometry));
        if (value > 0.f && visibleConnection(receiver, source.sample.surface))
            target = value;
    }
    result.add(source.sample, target, target * source.W * source.M, source.M, sampleNext1D(rng));
}

float3 resolveIndirect(Surface receiver, GIReservoir reservoir)
{
    float geometry;
    return reservoir.W > 0.f ? reservoir.W * contribution(receiver, reservoir.sample, geometry) : float3(0.f);
}

[numthreads(8, 8, 1)]
void generateCandidates(uint2 pixel : SV_DispatchThreadID)
{
    if (any(pixel >= gFrameDim))
        return;

    uint index = pixel.y * gFrameDim.x + pixel.x;
    Surface primary = loadPrimary(pixel);
    gPrimaryOut[index] = primary;
    GIReservoir result = {};
    float3 direct = 0.f;
    float3 status = 0.f;
    if (primary.valid != 0)
    {
        status = float3(0.1f, 0.23f, 0.7f); // No valid secondary surface.
        SampleGenerator rng = SampleGenerator(pixel, gFrame);
        float directionPdf;
        float3 direction = sampleDirection(primary.normal, rng, directionPdf);
        direct = directLighting(primary, rng);
        GISample candidate = {};
        float target = 0.f;
        float geometry = 0.f;
        if (dot(primary.geometricNormal, direction) > 0.f)
        {
            count(Stat::CandidateRay);
            candidate.surface = traceSurface(Ray(computeRayOrigin(primary.position, primary.geometricNormal), direction));
            if (candidate.surface.valid != 0)
            {
                candidate.direct = directLighting(candidate.surface, rng);
                float3 irradiance;
                bool hit = lookupCache(candidate.surface, irradiance);
                status = hit ? float3(0.1f, 0.85f, 0.38f) : float3(0.9f, 0.46f, 0.06f);
                candidate.setIrradiance(irradiance);
                if (!hit && gUpdateCache != 0)
                {
                    uint slot;
                    InterlockedAdd(gRequestCount[0], 1, slot);
                    if (slot < gRequestCapacity)
                        gRequestOut[slot] = candidate.surface;
                }
                target = luminance(contribution(primary, candidate, geometry));
            }
        }
        float areaPdf = directionPdf * geometry;
        result.add(candidate, target, areaPdf > 0.f ? target / areaPdf : 0.f, 1.f, 0.f);
        result.finalize();
    }
    gReservoirOut[index] = result;
    gDirect[pixel] = float4(direct, 1.f);
    if (gWriteSurfaceAlbedo != 0)
        gSurfaceAlbedo[pixel] = float4(primary.albedo, 1.f);
    if (gWriteSurfaceNormal != 0)
        gSurfaceNormal[pixel] = float4(primary.valid != 0 ? 0.5f + 0.5f * primary.normal : float3(0.f), 1.f);
    if (gWriteSurfaceDepth != 0)
    {
        float distance = length(primary.position - gScene.camera.getPosition());
        float depth = primary.valid != 0 ? saturate(log2(1.f + distance) / log2(1.f + 48.f * gCellSize)) : 0.f;
        gSurfaceDepth[pixel] = float4(float3(depth), 1.f);
    }
    if (gWriteCandidateIndirect != 0)
        gCandidateIndirect[pixel] = float4(resolveIndirect(primary, result), 1.f);
    if (gWriteCandidateStatus != 0)
        gCandidateStatus[pixel] = float4(status, 1.f);
}

[numthreads(8, 8, 1)]
void temporalResampling(uint2 pixel : SV_DispatchThreadID)
{
    if (any(pixel >= gFrameDim))
        return;

    uint index = pixel.y * gFrameDim.x + pixel.x;
    Surface primary = gPrimary[index];
    GIReservoir result = gInput[index];
    if (primary.valid != 0 && gUseTemporal != 0 && gHistoryValid != 0)
    {
        float2 previous = float2(pixel) + 0.5f + gMotion[pixel] * float2(gFrameDim);
        if (all(previous >= 0.f) && all(previous < float2(gFrameDim)))
        {
            uint2 prevPixel = uint2(previous);
            uint prevIndex = prevPixel.y * gFrameDim.x + prevPixel.x;
            float tolerance = max(gCellSize * 0.2f, 2.f * worldPixelScale(primary));
            if (compatible(primary, gPreviousPrimary[prevIndex], tolerance, gCellSize * 0.1f))
            {
                GIReservoir history = gHistory[prevIndex];
                if (history.M > 0.f)
                {
                    if (history.W > 0.f)
                    {
                        float3 irradiance;
                        lookupCache(history.sample.surface, irradiance);
                        history.sample.setIrradiance(irradiance);
                    }
                    SampleGenerator rng = SampleGenerator(pixel, gFrame ^ 0x1607u);
                    history.limitHistory(19.f);
                    merge(result, primary, history, rng);
                    result.finalize();
                }
            }
            else
                count(Stat::RejectedReprojection);
        }
        else
            count(Stat::RejectedReprojection);
    }
    gReservoirOut[index] = result;
    if (gWriteTemporalIndirect != 0)
        gTemporalIndirect[pixel] = float4(resolveIndirect(primary, result), 1.f);
    if (gWriteHistorySamples != 0)
        gHistorySamples[pixel] = float4(float3(saturate(result.M / 20.f)), 1.f);
}

[numthreads(8, 8, 1)]
void spatialAndShade(uint2 pixel : SV_DispatchThreadID)
{
    if (any(pixel >= gFrameDim))
        return;

    uint index = pixel.y * gFrameDim.x + pixel.x;
    Surface primary = gPrimary[index];
    if (gWriteSurfelOnly != 0 || gWriteSurfelDiscs != 0)
    {
        float3 irradiance = 0.f;
        float3 discColor = 0.f;
        if (gUseCache != 0 && primary.valid != 0)
            lookupSurfels(primary, gWriteSurfelDiscs != 0, irradiance, discColor);
        if (gWriteSurfelOnly != 0)
            gSurfelOnly[pixel] = float4(primary.albedo * (M_1_PI * irradiance), 1.f);
        if (gWriteSurfelDiscs != 0)
            gSurfelDiscs[pixel] = float4(discColor, 1.f);
    }
    if (gWriteColor == 0 && gWriteIndirect == 0)
        return;

    GIReservoir result = gInput[index];
    if (primary.valid != 0 && gUseSpatial != 0)
    {
        SampleGenerator rng = SampleGenerator(pixel, gFrame ^ 0x921au);
        static const int2 offsets[8] = {
            int2(-3, -1), int2(-1, -3), int2(1, -3), int2(3, -1),
            int2(3, 1), int2(1, 3), int2(-1, 3), int2(-3, 1)
        };
        uint start = uint(sampleNext1D(rng) * 8.f);
        float maxDistance = max(gCellSize, 6.f * worldPixelScale(primary));
        for (uint i = 0; i < 4; ++i)
        {
            int2 neighbor = int2(pixel) + offsets[(start + 2 * i) % 8];
            if (any(neighbor < 0) || any(neighbor >= int2(gFrameDim)))
                continue;

            uint otherIndex = uint(neighbor.y) * gFrameDim.x + uint(neighbor.x);
            if (compatible(primary, gPrimary[otherIndex], maxDistance, gCellSize * 0.25f))
                merge(result, primary, gInput[otherIndex], rng);
        }
        result.finalize();
    }

    float3 indirect = resolveIndirect(primary, result);
    if (gWriteIndirect != 0)
        gIndirect[pixel] = float4(indirect, 1.f);
    if (gWriteColor != 0)
        gColor[pixel] = float4(gDirectIn[pixel].xyz + indirect, 1.f);
}
