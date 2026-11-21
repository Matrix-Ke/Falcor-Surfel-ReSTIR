#include "Utils/Math/MathConstants.slangh"

struct Surface
{
    float3 position;
    uint instance;
    float3 geometricNormal;
    uint valid;
    float3 normal;
    float3 albedo;
};

struct GISample
{
    Surface surface;
    float3 direct;
    float3 cached;

    float3 radiance()
    {
        return direct + cached;
    }

    [mutating]
    void setIrradiance(float3 irradiance)
    {
        cached = surface.albedo * (M_1_PI * irradiance);
    }
};

// The area Jacobian is also used to convert the initial direction PDF.
float3 contribution(Surface receiver, GISample candidate, out float geometry)
{
    geometry = 0.f;
    float3 offset = candidate.surface.position - receiver.position;
    float distanceSquared = dot(offset, offset);
    if (distanceSquared <= 1e-12f)
        return 0.f;

    float3 direction = offset * rsqrt(distanceSquared);
    float cosine = dot(receiver.normal, direction);
    float hitCosine = dot(candidate.surface.geometricNormal, -direction);
    if (cosine <= 0.f || hitCosine <= 0.f || dot(receiver.geometricNormal, direction) <= 0.f)
        return 0.f;

    geometry = hitCosine / distanceSquared;
    return candidate.radiance() * receiver.albedo * (M_1_PI * cosine * geometry);
}

struct GIReservoir
{
    GISample sample;
    float weightSum;
    float M;
    float W;
    float selectedTarget;

    [mutating]
    void add(GISample candidate, float target, float weight, float count, float random)
    {
        M += count;
        if (!(weight > 0.f && isfinite(weight)))
            return;

        weightSum += weight;
        if (random * weightSum < weight)
        {
            sample = candidate;
            selectedTarget = target;
        }
    }

    [mutating]
    void finalize()
    {
        W = M > 0.f && selectedTarget > 0.f ? (weightSum / M) / selectedTarget : 0.f;
    }

    [mutating]
    void limitHistory(float limit)
    {
        if (M > limit)
        {
            weightSum *= limit / M;
            M = limit;
        }
    }
};
