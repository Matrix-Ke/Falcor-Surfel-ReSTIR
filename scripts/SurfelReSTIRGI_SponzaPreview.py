from pathlib import Path
from runpy import run_path


render_graph = run_path(str(Path(__file__).with_name("SurfelReSTIRGI.py")))["render_graph_SurfelReSTIRGI"]
graph = render_graph(exposure_compensation=2.5)
graph.createPass("SurfelOnly", "ToneMapper", {"autoExposure": False, "exposureCompensation": 2.5})
graph.addEdge("GI.surfelOnly", "SurfelOnly.src")
graph.markOutput("SurfelOnly.dst")
graph.markOutput("GI.surfelDiscs")
m.addGraph(graph)
