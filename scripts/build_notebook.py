"""Genera y ejecuta el notebook real de la pre-entrega 6."""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

root=Path(__file__).resolve().parents[1]
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={"display_name":"Python 3.12","language":"python","name":"python3"}
nb.cells=[
 nbformat.v4.new_markdown_cell("# Pre-entrega 6 — Orquestador multiagente\n\n## Objetivo\nDemostrar delegacion real a un investigador y un analista, con herramientas funcionales y un supervisor que exige validacion antes de finalizar.\n\nEl corpus de cuatro documentos es original y educativo. La ejecucion usa DeepSeek Flash real, embeddings ONNX existentes, Chroma y Phoenix. No guarda claves ni descarga modelos."),
 nbformat.v4.new_markdown_cell("## Configuracion\nEjecutar desde la raiz con requirements.lock.txt ya instalado. La credencial autorizada llega solamente al entorno del proceso; nunca se muestra ni se incorpora al notebook. Los embeddings ya deben estar en cache. No hace falta que FastAPI este activa para este ejemplo."),
 nbformat.v4.new_code_cell("from app.llm import ModelGateway\nfrom app.rag import HybridKnowledgeBase\nfrom app.graph import build_graph\nfrom langchain_core.messages import HumanMessage\n\ngateway = ModelGateway()\nawait gateway.validate_environment()\nknowledge = HybridKnowledgeBase(gateway)\nawait knowledge.setup()\ngraph = build_graph(gateway, knowledge)\nprint('Documentos fragmentados:', len(knowledge.chunks))"),
 nbformat.v4.new_markdown_cell("## Ejecucion y delegacion\nLa pregunta requiere recuperar la politica de persistencia y analizar como se retoma una ejecucion. Se guardan decisiones operativas y aportes, no razonamiento privado del modelo."),
 nbformat.v4.new_code_cell("query = 'Como se conserva un trabajo cuando reinicia FastAPI y como lo valida el supervisor?'\nresult = await graph.ainvoke({'query': query, 'messages': [HumanMessage(content=query)], 'contributions': [], 'usages': [], 'routing_log': [], 'steps': 0})\nfor decision in result['routing_log']:\n    print(decision['step'], decision['selected'], '-', decision['reason'])\nprint('\\nAportes y herramientas')\nfor contribution in result['contributions']:\n    print(contribution['agent'], '->', contribution['tool'])"),
 nbformat.v4.new_markdown_cell("## Resultado validado\nLas referencias deben existir en los fragmentos recuperados. Una respuesta afirmativa sin referencias no puede superar el validador."),
 nbformat.v4.new_code_cell("from pprint import pprint\npprint(result['final'])\nassert result['final']['validation_passed']\nassert set(result['final']['contributors']) == {'researcher', 'analyst'}\nassert all(r['id'] in {d['id'] for d in result['evidence']} for r in result['final']['references'])"),
 nbformat.v4.new_markdown_cell("## Tokens y costo\nLos tokens LLM son informados por DeepSeek, incluyendo intentos invalidos antes de un reintento exitoso. Los embeddings ONNX no tienen un contador comparable; no se inventa. El costo usa tarifas Flash OFF-PEAK verificadas el 26 de septiembre de 2026; es una estimacion y no una factura."),
 nbformat.v4.new_code_cell("from app.observability import usage_cost\nprint('Tokens entrada:', sum(u.get('input_tokens', 0) for u in result['usages']))\nprint('Tokens salida:', sum(u.get('output_tokens', 0) for u in result['usages']))\nprint('Costo de esta ejecucion USD:', round(sum(usage_cost(u) for u in result['usages']), 6))\nawait gateway.close()"),
 nbformat.v4.new_markdown_cell("## Conclusiones\nEl supervisor delega segun los artefactos disponibles. El investigador consulta la base hibrida y el analista verifica el esquema y las fuentes. El nodo final valida el contrato y produce una respuesta o una limitacion. Los ciclos estan acotados a una ronda de refinamiento y ocho decisiones.\n\nLas pruebas adicionales de concurrencia, interrupcion y reinicio estan en tests/test_system.py y evidence/benchmark.json."),
]
nbformat.validate(nb)
manager=KernelManager(kernel_name="python3")
manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
client=NotebookClient(nb,timeout=600,resources={"metadata":{"path":str(root)}})
client.km=manager
client.execute()
output=root/"notebooks"
output.mkdir(exist_ok=True)
nbformat.write(nb,output/"preentrega6_demo.ipynb")
html,_=HTMLExporter().from_notebook_node(nb)
(output/"preentrega6_demo.html").write_text(html,encoding="utf-8")
print("Notebook ejecutado y validado; guardado en notebooks/preentrega6_demo.ipynb")
lcel = nbformat.v4.new_notebook()
lcel.metadata.kernelspec = nb.metadata.kernelspec
lcel.cells = [
    nbformat.v4.new_markdown_cell("# Modulo 2, unidad 1 - LCEL asincrono\n\nLa cadena usa ChatPromptTemplate, ChatOpenAI y StrOutputParser conectados con |. ChatOpenAI se usa como adaptador compatible con DeepSeek: no se utiliza una clave de OpenAI. La credencial llega al entorno sin guardarse ni mostrarse. El resultado es texto plano, no AIMessage. Codigo completo en exercises/m2u1_lcel.py."),
    nbformat.v4.new_code_cell("from exercises.m2u1_lcel import responder\nrespuesta = await responder('Por que hay que usar await con chain.ainvoke?')\nassert isinstance(respuesta, str) and respuesta.strip()\nprint(type(respuesta).__name__)\nprint(respuesta)"),
    nbformat.v4.new_markdown_cell("## Conclusion\n\nEl diccionario pregunta llena la variable del prompt. await espera la ejecucion asincrona de la cadena. StrOutputParser extrae el contenido textual. Se usan roles system y human y no se codifica manualmente el flujo principal. La llamada real se dirige solo al HTTPS oficial de DeepSeek."),
]
manager_lcel = KernelManager(kernel_name="python3")
manager_lcel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client_lcel = NotebookClient(lcel, timeout=180, resources={"metadata": {"path": str(root)}})
client_lcel.km = manager_lcel
client_lcel.execute()
nbformat.validate(lcel)
nbformat.write(lcel, output / "m2u1_lcel_demo.ipynb")
print("Demostracion LCEL ejecutada sin errores")
