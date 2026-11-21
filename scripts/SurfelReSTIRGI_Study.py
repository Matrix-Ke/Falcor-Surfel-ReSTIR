from pathlib import Path
from runpy import run_path


create_gi_graph = run_path(str(Path(__file__).with_name("SurfelReSTIRGI.py")))["render_graph_SurfelReSTIRGI"]


def render_graph_SurfelReSTIRGI_Study(graph=None, exposure_compensation=2.5, properties=None):
    graph = create_gi_graph(graph, properties=properties, display=False)
    graph.name = "SurfelReSTIRGI Study"
    for channel in ("color", "direct", "indirect"):
        graph.unmarkOutput(f"GI.{channel}")

    # Open on the final image, then list the stages in data-flow order.
    stages = (
        ("Final", "color", True),
        ("Albedo", "surfaceAlbedo", False),
        ("Normals", "surfaceNormal", False),
        ("Depth", "surfaceDepth", False),
        ("Direct", "direct", True),
        ("SurfelDiscs", "surfelDiscs", False),
        ("SurfelOnly", "surfelOnly", True),
        ("CandidateStatus", "candidateStatus", False),
        ("CandidateIndirect", "candidateIndirect", True),
        ("HistorySamples", "historySamples", False),
        ("TemporalIndirect", "temporalIndirect", True),
        ("SpatialIndirect", "indirect", True),
    )
    for label, channel, tone_map in stages:
        source = f"GI.{channel}"
        if tone_map:
            graph.createPass(label, "ToneMapper", {
                "autoExposure": False,
                "exposureCompensation": exposure_compensation,
            })
            graph.addEdge(source, f"{label}.src")
            graph.markOutput(f"{label}.dst")
        else:
            graph.markOutput(source)
    return graph


if "m" in globals():
    m.addGraph(render_graph_SurfelReSTIRGI_Study())
