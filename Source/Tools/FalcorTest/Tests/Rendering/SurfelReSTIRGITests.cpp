#include "Testing/UnitTest.h"

#include <cmath>

namespace Falcor
{
GPU_TEST(SurfelReSTIRGIWeights)
{
    constexpr uint32_t kTrials = 1024;
    ctx.createProgram("Tests/Rendering/SurfelReSTIRGITests.cs.slang", "testWeights");
    ctx.allocateStructuredBuffer("gResults", kTrials);
    ctx.runProgram(kTrials);

    const auto results = ctx.readBuffer<float4>("gResults");
    uint32_t uncachedWins = 0;
    uint32_t cachedWins = 0;
    for (const auto& result : results)
    {
        uncachedWins += result.x > 0.f;
        cachedWins += result.y > 0.f;
    }
    EXPECT_EQ(uncachedWins, kTrials / 2);
    EXPECT_EQ(cachedWins, 3 * kTrials / 4);
    EXPECT_LT(std::abs(results[0].z - 1.f), 1e-6f);
    EXPECT_LT(std::abs(results[0].w - 3.f), 1e-6f);
}
} // namespace Falcor
