add_plugin(SurfelReSTIRGI)

target_sources(SurfelReSTIRGI PRIVATE
    SurfelReSTIRGI.cpp
    SurfelReSTIRGI.cs.slang
    Reservoir.slang
)

target_copy_shaders(SurfelReSTIRGI RenderPasses/SurfelReSTIRGI)
target_source_group(SurfelReSTIRGI "RenderPasses")
