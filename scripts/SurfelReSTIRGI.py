from falcor import *


def render_graph_SurfelReSTIRGI(graph=None, properties=None, display=True, exposure_compensation=0.0):
    if graph is None:
        graph = RenderGraph("SurfelReSTIRGI")
    graph.createPass("VBuffer", "VBufferRT", {
        "samplePattern": "Center",
        "sampleCount": 1,
        "useAlphaTest": False,
        "useDOF": False,
    })
    graph.createPass("GI", "SurfelReSTIRGI", properties or {})
    for channel in ("vbuffer", "viewW", "mvec"):
        graph.addEdge(f"VBuffer.{channel}", f"GI.{channel}")

    if display:
        graph.createPass("ToneMapper", "ToneMapper", {
            "autoExposure": False,
            "exposureCompensation": exposure_compensation,
        })
        graph.addEdge("GI.color", "ToneMapper.src")
        graph.markOutput("ToneMapper.dst")
    else:
        graph.markOutput("GI.color")
    graph.markOutput("GI.direct")
    graph.markOutput("GI.indirect")
    return graph


if "m" in globals():
    m.addGraph(render_graph_SurfelReSTIRGI())
