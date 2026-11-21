#include "Utils/Math/MathConstants.slangh"

import RenderPasses.SurfelReSTIRGI.Reservoir;
import Utils.Color.ColorHelpers;

RWStructuredBuffer<float4> gResults;

[numthreads(64, 1, 1)]
void testWeights(uint index : SV_DispatchThreadID)
{
    Surface receiver = {};
    receiver.normal = receiver.geometricNormal = float3(0, 0, 1);
    receiver.albedo = 0.5f;

    GISample first = {};
    first.direct = 2.f;
    first.surface.position = float3(0, 0, 2);
    first.surface.geometricNormal = float3(0, 0, -1);
    first.surface.albedo = 0.25f;
    GISample second = first;
    // A different endpoint with the same geometry and direct target.
    first.surface.instance = 1;
    second.surface.instance = 2;

    float geometry;
    float3 directContribution = contribution(receiver, first, geometry);
    float areaPdf = M_1_PI * geometry;
    float directTarget = luminance(directContribution);
    float random = (index + 0.5f) / 1024.f;
    GIReservoir uncached = {};
    uncached.add(first, directTarget, directTarget / areaPdf, 1.f, 0.f);
    uncached.add(second, directTarget, directTarget / areaPdf, 1.f, random);

    second.setIrradiance(16.f * M_PI);
    float3 cachedContribution = contribution(receiver, second, geometry);
    float cachedTarget = luminance(cachedContribution);
    GIReservoir combined = {};
    combined.add(first, directTarget, directTarget / areaPdf, 1.f, 0.f);
    combined.add(second, cachedTarget, cachedTarget / areaPdf, 1.f, random);

    GIReservoir single = {};
    single.add(first, directTarget, directTarget / areaPdf, 1.f, 0.f);
    single.finalize();
    float singleDirect = (single.W * directContribution).x;
    single = {};
    single.add(second, cachedTarget, cachedTarget / areaPdf, 1.f, 0.f);
    single.finalize();
    float singleCached = (single.W * cachedContribution).x;

    gResults[index] = float4(uncached.sample.surface.instance == 2, combined.sample.surface.instance == 2, singleDirect, singleCached);
}
