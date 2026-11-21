#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Math/AABB.h"
#include "Utils/Sampling/SampleGenerator.h"

#include <algorithm>
#include <array>
#include <cmath>

using namespace Falcor;

namespace
{
constexpr uint32_t kBucketCount = 8192;
constexpr uint32_t kCapacity = kBucketCount * 4;
constexpr uint32_t kRequestCapacity = 8192;
constexpr uint32_t kTileSize = 8;
constexpr std::array kOutputs = {
    "direct", "indirect", "color", "surfelOnly", "surfelDiscs",
    "surfaceAlbedo", "surfaceNormal", "surfaceDepth", "candidateIndirect", "candidateStatus", "temporalIndirect", "historySamples"
};
constexpr std::array kStats = {
    "Candidate rays", "Reconnection rays", "Surfel update rays", "Cache queries", "Cache hits", "Direct shadow rays",
    "Occluded reconnections", "Rejected reprojections"
};

class SurfelReSTIRGI : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SurfelReSTIRGI, "SurfelReSTIRGI", "Diffuse ReSTIR GI with a world-space surfel cache.");

    static ref<SurfelReSTIRGI> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<SurfelReSTIRGI>(pDevice, props);
    }

    static void registerBindings(pybind11::module& module)
    {
        pybind11::class_<SurfelReSTIRGI, RenderPass, ref<SurfelReSTIRGI>> pass(module, "SurfelReSTIRGI");
        pass.def("getStatistics", &SurfelReSTIRGI::getStatistics);
        pass.def_readwrite("freezeCache", &SurfelReSTIRGI::mFreezeCache);
    }

    SurfelReSTIRGI(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
    {
        mUseCache = props.get("useCache", true);
        mFreezeCache = props.get("freezeCache", false);
        mUseTemporal = props.get("useTemporal", true);
        mUseSpatial = props.get("useSpatial", true);
        mCollectStats = props.get("collectStats", false);
        mSurfelRayBudget = std::clamp(props.get("surfelRayBudget", 16384u), 1u, kCapacity);
        mCellSize = props.get("cellSize", 0.f);
        FALCOR_CHECK(mCellSize >= 0.f && std::isfinite(mCellSize), "cellSize must be finite and nonnegative.");
        FALCOR_CHECK(
            pDevice->isShaderModelSupported(ShaderModel::SM6_5) &&
                pDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1),
            "SurfelReSTIRGI requires Shader Model 6.5 and Raytracing Tier 1.1."
        );
        mpSampleGenerator = SampleGenerator::create(pDevice, SAMPLE_GENERATOR_DEFAULT);
    }

    Properties getProperties() const override
    {
        Properties props;
        props["useCache"] = mUseCache;
        props["freezeCache"] = mFreezeCache;
        props["useTemporal"] = mUseTemporal;
        props["useSpatial"] = mUseSpatial;
        props["collectStats"] = mCollectStats;
        props["surfelRayBudget"] = mSurfelRayBudget;
        props["cellSize"] = mCellSize;
        return props;
    }

    RenderPassReflection reflect(const CompileData& compileData) override
    {
        RenderPassReflection reflection;
        reflection.addInput("vbuffer", "Packed visibility buffer").bindFlags(ResourceBindFlags::ShaderResource);
        reflection.addInput("viewW", "World-space view direction").bindFlags(ResourceBindFlags::ShaderResource);
        reflection.addInput("mvec", "Motion from current to previous UV").bindFlags(ResourceBindFlags::ShaderResource);
        for (const auto& name : kOutputs)
        {
            reflection.addOutput(name, "Radiance or diagnostic view")
                .texture2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y)
                .format(ResourceFormat::RGBA32Float)
                .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        }
        return reflection;
    }

    void setScene(RenderContext*, const ref<Scene>& pScene) override
    {
        mpScene = pScene;
        invalidatePrograms();
    }

    void renderUI(Gui::Widgets& widget) override
    {
        if (widget.checkbox("Surfel cache", mUseCache))
        {
            mResetCache = true;
            mOptionsChanged = true;
        }
        widget.checkbox("Freeze surfel cache", mFreezeCache);
        mOptionsChanged |= widget.checkbox("Temporal reuse", mUseTemporal);
        mOptionsChanged |= widget.checkbox("Spatial reuse", mUseSpatial);
        mOptionsChanged |= widget.var("Surfel ray budget", mSurfelRayBudget, 1u, kCapacity);
        if (widget.button("Reset cache and history"))
        {
            mResetCache = true;
            mOptionsChanged = true;
        }
        widget.checkbox("Collect ray and cache statistics", mCollectStats);
        if (mCollectStats)
        {
            widget.tooltip("Reads counters back synchronously. Disable this for GPU timing.");
            for (size_t i = 0; i < kStats.size(); ++i)
                widget.text(fmt::format("{}: {}", kStats[i], mStats[i]));
            if (mStats[3] != 0)
                widget.text(fmt::format("Cache hit rate: {:.1f}%", 100.0 * mStats[4] / mStats[3]));
            widget.text(fmt::format("Internal buffers: {:.1f} MiB", mBufferBytes / (1024.0 * 1024.0)));
        }
    }

    void execute(RenderContext* pContext, const RenderData& renderData) override
    {
        if (!mpScene)
        {
            for (const auto& name : kOutputs)
                if (auto texture = renderData.getTexture(name))
                    pContext->clearUAV(texture->getUAV().get(), float4(0.f, 0.f, 0.f, 1.f));
            return;
        }

        const auto& vbuffer = renderData.getTexture("vbuffer");
        const uint2 dim(vbuffer->getWidth(), vbuffer->getHeight());
        const auto checkDimensions = [&](const char* name)
        {
            const auto& texture = renderData.getTexture(name);
            FALCOR_CHECK(!texture || (texture->getWidth() == dim.x && texture->getHeight() == dim.y), "{} must match vbuffer dimensions.", name);
        };
        checkDimensions("viewW");
        checkDimensions("mvec");
        for (const auto& name : kOutputs)
            checkDimensions(name);

        using UpdateFlags = IScene::UpdateFlags;
        const auto updates = mpScene->getUpdates();
        if ((updates & UpdateFlags::RecompileNeeded) != UpdateFlags::None)
            invalidatePrograms();

        const auto cacheChanges = UpdateFlags::GeometryChanged | UpdateFlags::GeometryMoved | UpdateFlags::MeshesChanged |
            UpdateFlags::MaterialsChanged | UpdateFlags::LightsMoved | UpdateFlags::LightIntensityChanged |
            UpdateFlags::LightPropertiesChanged | UpdateFlags::LightCountChanged | UpdateFlags::RenderSettingsChanged;
        if ((updates & cacheChanges) != UpdateFlags::None)
            mResetCache = true;
        if (mResetCache || mOptionsChanged || is_set(updates, UpdateFlags::CameraSwitched))
            mHistoryValid = false;

        if (mOptionsChanged)
        {
            auto& dictionary = renderData.getDictionary();
            auto flags = dictionary.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
            dictionary[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::RenderOptionsChanged;
            mOptionsChanged = false;
        }

        if (!mpCandidates)
            createPrograms();
        if (!mpInitial || any(mFrameDim != dim))
            allocateScreenBuffers(dim);

        auto direct = renderData.getTexture("direct");
        auto indirect = renderData.getTexture("indirect");
        auto color = renderData.getTexture("color");
        auto surfelOnly = renderData.getTexture("surfelOnly");
        auto surfelDiscs = renderData.getTexture("surfelDiscs");
        auto albedo = renderData.getTexture("surfaceAlbedo");
        auto normal = renderData.getTexture("surfaceNormal");
        auto depth = renderData.getTexture("surfaceDepth");
        auto candidate = renderData.getTexture("candidateIndirect");
        auto candidateStatus = renderData.getTexture("candidateStatus");
        auto temporal = renderData.getTexture("temporalIndirect");
        auto historySamples = renderData.getTexture("historySamples");
        if (!direct)
        {
            if (!mpDirectScratch)
                mpDirectScratch = mpDevice->createTexture2D(
                    dim.x, dim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
                );
            direct = mpDirectScratch;
        }

        const auto& bounds = mpScene->getSceneBounds();
        const float sceneDiameter = bounds.valid() ? 2.f * bounds.radius() : 0.f;
        const float3 gridOrigin = bounds.valid() ? bounds.minPoint : float3(0.f);
        const float cellSize = mCellSize > 0.f ? mCellSize : std::max(1e-4f, sceneDiameter / 48.f);
        const uint2 tileDim((dim.x + kTileSize - 1) / kTileSize, (dim.y + kTileSize - 1) / kTileSize);
        const uint32_t next = mPrevious ^ 1;
        const uint32_t nextCache = mPreviousCache ^ 1;
        const bool updateCache = mUseCache && !mFreezeCache;
        const bool allowSurfelEviction = is_set(updates, UpdateFlags::CameraMoved);
        const uint32_t publishedCache = updateCache ? nextCache : mPreviousCache;

        const auto dispatch = [&](const ref<ComputePass>& pass, const char* name, uint3 threads, auto bind)
        {
            FALCOR_PROFILE(pContext, name);
            auto root = pass->getRootVar();
            auto frame = root["PerFrame"];
            frame["gFrameDim"] = dim;
            frame["gTileDim"] = tileDim;
            frame["gFrame"] = mFrame;
            frame["gBucketCount"] = kBucketCount;
            frame["gRequestCapacity"] = kRequestCapacity;
            frame["gUpdateBudget"] = mSurfelRayBudget;
            frame["gUpdateStart"] = mUpdateStart;
            frame["gUseCache"] = uint32_t(mUseCache);
            frame["gUpdateCache"] = uint32_t(updateCache);
            frame["gAllowSurfelEviction"] = uint32_t(allowSurfelEviction);
            frame["gUseTemporal"] = uint32_t(mUseTemporal);
            frame["gUseSpatial"] = uint32_t(mUseSpatial);
            frame["gHistoryValid"] = uint32_t(mHistoryValid);
            frame["gCollectStats"] = uint32_t(mCollectStats);
            frame["gWriteIndirect"] = uint32_t(bool(indirect));
            frame["gWriteColor"] = uint32_t(bool(color));
            frame["gWriteSurfelOnly"] = uint32_t(bool(surfelOnly));
            frame["gWriteSurfelDiscs"] = uint32_t(bool(surfelDiscs));
            frame["gWriteSurfaceAlbedo"] = uint32_t(bool(albedo));
            frame["gWriteSurfaceNormal"] = uint32_t(bool(normal));
            frame["gWriteSurfaceDepth"] = uint32_t(bool(depth));
            frame["gWriteCandidateIndirect"] = uint32_t(bool(candidate));
            frame["gWriteCandidateStatus"] = uint32_t(bool(candidateStatus));
            frame["gWriteTemporalIndirect"] = uint32_t(bool(temporal));
            frame["gWriteHistorySamples"] = uint32_t(bool(historySamples));
            frame["gUseAnalyticLights"] = uint32_t(mpScene->useAnalyticLights());
            frame["gCellSize"] = cellSize;
            frame["gGridOrigin"] = gridOrigin;
            bind(root);
            pass->execute(pContext, threads);
        };
        const auto bindScene = [&](const ShaderVar& root)
        {
            mpScene->bindShaderDataForRaytracing(pContext, root["gScene"]);
        };
        const auto bindSurfaceInputs = [&](const ShaderVar& root)
        {
            root["gVBuffer"] = vbuffer;
            root["gViewW"] = renderData.getTexture("viewW");
        };

        if (mCollectStats)
            pContext->clearUAV(mpStats->getUAV().get(), uint4(0));
        if (mResetCache)
        {
            dispatch(mpReset, "Reset surfel cache", uint3(kCapacity, 1, 1), [&](const ShaderVar& root)
            {
                root["gSurfelOut"] = mpSurfels;
                root["gCacheOut"] = mCache[mPreviousCache];
            });
            pContext->clearUAV(mpRequestCount->getUAV().get(), uint4(0));
            mUpdateStart = 0;
            mResetCache = false;
        }

        if (updateCache)
        {
            pContext->clearUAV(mpProposals->getUAV().get(), uint4(~0u));
            dispatch(mpPropose, "Propose surfels", uint3(tileDim.x * tileDim.y + kRequestCapacity, 1, 1), [&](const ShaderVar& root)
            {
                bindScene(root);
                bindSurfaceInputs(root);
                root["gProposals"] = mpProposals;
                root["gRequestCount"] = mpRequestCount;
                root["gRequests"] = mpRequests;
            });
            dispatch(mpUpdate, "Trace surfel irradiance", uint3(kCapacity, 1, 1), [&](const ShaderVar& root)
            {
                bindScene(root);
                mpSampleGenerator->bindShaderData(root);
                root["gStats"] = mpStats;
                root["gSurfels"] = mpSurfels;
                root["gCache"] = mCache[mPreviousCache];
                root["gCacheOut"] = mCache[nextCache];
            });
            // Allocate only after irradiance update has finished reading the old slot identities.
            dispatch(mpAllocate, "Allocate surfels", uint3(kBucketCount, 1, 1), [&](const ShaderVar& root)
            {
                bindScene(root);
                bindSurfaceInputs(root);
                root["gProposals"] = mpProposals;
                root["gRequests"] = mpRequests;
                root["gSurfelOut"] = mpSurfels;
                root["gCacheOut"] = mCache[nextCache];
            });
            pContext->clearUAV(mpRequestCount->getUAV().get(), uint4(0));
        }

        dispatch(mpCandidates, "Generate GI candidates", uint3(dim, 1), [&](const ShaderVar& root)
        {
            bindScene(root);
            bindSurfaceInputs(root);
            mpSampleGenerator->bindShaderData(root);
            root["gStats"] = mpStats;
            root["gSurfels"] = mpSurfels;
            root["gCache"] = mCache[publishedCache];
            root["gRequestCount"] = mpRequestCount;
            root["gRequestOut"] = mpRequests;
            root["gPrimaryOut"] = mPrimary[next];
            root["gReservoirOut"] = mpInitial;
            root["gDirect"] = direct;
            root["gSurfaceAlbedo"] = albedo;
            root["gSurfaceNormal"] = normal;
            root["gSurfaceDepth"] = depth;
            root["gCandidateIndirect"] = candidate;
            root["gCandidateStatus"] = candidateStatus;
        });
        dispatch(mpTemporal, "Temporal GI reuse", uint3(dim, 1), [&](const ShaderVar& root)
        {
            bindScene(root);
            mpSampleGenerator->bindShaderData(root);
            root["gStats"] = mpStats;
            root["gSurfels"] = mpSurfels;
            root["gCache"] = mCache[publishedCache];
            root["gMotion"] = renderData.getTexture("mvec");
            root["gPrimary"] = mPrimary[next];
            root["gPreviousPrimary"] = mPrimary[mPrevious];
            root["gHistory"] = mTemporal[mPrevious];
            root["gInput"] = mpInitial;
            root["gReservoirOut"] = mTemporal[next];
            root["gTemporalIndirect"] = temporal;
            root["gHistorySamples"] = historySamples;
        });
        dispatch(mpShade, "Spatial GI reuse and shade", uint3(dim, 1), [&](const ShaderVar& root)
        {
            bindScene(root);
            mpSampleGenerator->bindShaderData(root);
            root["gStats"] = mpStats;
            root["gPrimary"] = mPrimary[next];
            root["gInput"] = mTemporal[next];
            root["gDirectIn"] = direct;
            root["gIndirect"] = indirect;
            root["gColor"] = color;
            root["gSurfelOnly"] = surfelOnly;
            root["gSurfelDiscs"] = surfelDiscs;
            root["gSurfels"] = mpSurfels;
            root["gCache"] = mCache[publishedCache];
        });

        if (mCollectStats)
            pContext->readBuffer(mpStats.get(), mStats.data(), 0, sizeof(mStats));
        if (updateCache)
        {
            mPreviousCache = nextCache;
            mUpdateStart = (mUpdateStart + mSurfelRayBudget) % kCapacity;
        }
        mPrevious = next;
        mHistoryValid = true;
        ++mFrame;
    }

private:
    pybind11::dict getStatistics() const
    {
        pybind11::dict result;
        if (mCollectStats)
        {
            for (size_t i = 0; i < kStats.size(); ++i)
                result[kStats[i]] = mStats[i];
            result["Internal buffer bytes"] = mBufferBytes;
        }
        return result;
    }

    void invalidatePrograms()
    {
        mpCandidates = nullptr;
        mpInitial = nullptr;
        mResetCache = true;
        mHistoryValid = false;
        mFrame = 0;
    }

    void createPrograms()
    {
        const auto create = [&](const char* entry)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary("RenderPasses/SurfelReSTIRGI/SurfelReSTIRGI.cs.slang").csEntry(entry);
            desc.addTypeConformances(mpScene->getTypeConformances());
            desc.setShaderModel(ShaderModel::SM6_5);
            DefineList defines = mpScene->getSceneDefines();
            defines.add(mpSampleGenerator->getDefines());
            defines.add("PAD_RANDOM_NUMBERS", "0");
            return ComputePass::create(mpDevice, desc, defines, true);
        };
        mpReset = create("resetCache");
        mpPropose = create("proposeSurfels");
        mpUpdate = create("updateIrradiance");
        mpAllocate = create("allocateSurfels");
        mpCandidates = create("generateCandidates");
        mpTemporal = create("temporalResampling");
        mpShade = create("spatialAndShade");
        mpSurfels = mpDevice->createStructuredBuffer(mpReset->getRootVar()["gSurfelOut"], kCapacity);
        for (auto& cache : mCache)
            cache = mpDevice->createStructuredBuffer(mpReset->getRootVar()["gCacheOut"], kCapacity);
        mpRequests = mpDevice->createStructuredBuffer(mpCandidates->getRootVar()["gRequestOut"], kRequestCapacity);
        mpRequestCount = mpDevice->createTypedBuffer<uint32_t>(1);
        mpProposals = mpDevice->createTypedBuffer<uint32_t>(kBucketCount);
        mpStats = mpDevice->createTypedBuffer<uint32_t>(kStats.size());
    }

    void allocateScreenBuffers(uint2 dim)
    {
        auto root = mpCandidates->getRootVar();
        const uint32_t count = dim.x * dim.y;
        mpInitial = mpDevice->createStructuredBuffer(root["gReservoirOut"], count);
        for (auto& buffer : mTemporal)
            buffer = mpDevice->createStructuredBuffer(root["gReservoirOut"], count);
        for (auto& buffer : mPrimary)
            buffer = mpDevice->createStructuredBuffer(root["gPrimaryOut"], count);

        mBufferBytes = mpInitial->getSize() + mpSurfels->getSize() + mpRequests->getSize() +
            mpRequestCount->getSize() + mpProposals->getSize() + mpStats->getSize();
        for (size_t i = 0; i < 2; ++i)
            mBufferBytes += mTemporal[i]->getSize() + mPrimary[i]->getSize() + mCache[i]->getSize();
        mFrameDim = dim;
        mpDirectScratch = nullptr;
        mHistoryValid = false;
    }

    ref<Scene> mpScene;
    ref<Texture> mpDirectScratch;
    ref<SampleGenerator> mpSampleGenerator;
    ref<ComputePass> mpReset, mpPropose, mpUpdate, mpAllocate, mpCandidates, mpTemporal, mpShade;
    ref<Buffer> mpSurfels, mpRequests, mpRequestCount, mpProposals, mpStats, mpInitial;
    std::array<ref<Buffer>, 2> mCache, mTemporal, mPrimary;
    std::array<uint32_t, kStats.size()> mStats{};
    size_t mBufferBytes = 0;
    uint2 mFrameDim = {};
    uint32_t mFrame = 0;
    uint32_t mPrevious = 0;
    uint32_t mPreviousCache = 0;
    uint32_t mUpdateStart = 0;
    uint32_t mSurfelRayBudget = 0;
    float mCellSize = 0.f;
    bool mUseCache = true;
    bool mFreezeCache = false;
    bool mUseTemporal = true;
    bool mUseSpatial = true;
    bool mCollectStats = false;
    bool mOptionsChanged = false;
    bool mHistoryValid = false;
    bool mResetCache = true;
};
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SurfelReSTIRGI>();
    ScriptBindings::registerBinding(SurfelReSTIRGI::registerBindings);
}
