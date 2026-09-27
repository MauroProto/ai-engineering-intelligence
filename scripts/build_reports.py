"""PDF de estudiante, sin portadas ilustradas ni elementos decorativos."""
from html import escape
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf"
OUT.mkdir(parents=True, exist_ok=True)
normal = ParagraphStyle("normal", fontName="Times-Roman", fontSize=11, leading=15,
                        spaceAfter=8, alignment=TA_LEFT)
h1 = ParagraphStyle("heading", parent=normal, fontName="Times-Bold", fontSize=14,
                    leading=18, spaceBefore=12, spaceAfter=9, keepWithNext=True)
code = ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10.5)


def paragraph(text):
    return Paragraph(escape(text), normal)


def footer(canvas, doc):
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(60, 35, "AI Engineering - Mauro Tomas Proto Cassina")
    canvas.drawRightString(A4[0]-60,35,str(doc.page))


def from_markdown(source):
    story=[]
    for block in source.split("\n\n"):
        block=" ".join(block.strip().splitlines())
        if not block: continue
        if block.startswith("#"):
            story.append(Paragraph(escape(block.lstrip("# ")),h1))
        else:
            story.append(paragraph(block))
    return story


def save(name, story):
    SimpleDocTemplate(str(OUT/name), pagesize=A4, leftMargin=60, rightMargin=60,
                      topMargin=55, bottomMargin=55,
                      title=name[:-4], author="Mauro Tomas Proto Cassina").build(
                      story,onFirstPage=footer,onLaterPages=footer)


def main():
    evaluation=(ROOT/"docs/evaluacion.md").read_text()
    jobs=json.loads((ROOT/"evidence/benchmark.json").read_text())["jobs"]
    obs=from_markdown(evaluation)
    obs += [PageBreak(),Paragraph("Anexo - Resultados de las cinco consultas",h1)]
    for index,job in enumerate(jobs,1):
        obs += [Paragraph(f"Consulta {index}",h1),paragraph(job["request"]["query"]),
                paragraph(job["result"]["answer"]),
                paragraph(f"Latencia desde encolado {job['latency_seconds']:.6f} s. "
                          f"Costo estimado {job['cost_usd']:.9f} USD. "
                          f"Trace ID {job['trace_id']}."),
                paragraph("Fuentes recuperadas: "+", ".join(sorted({r["source"] for r in job["result"]["references"]}))),
                paragraph("Limites informados: "+job["result"]["limitations"])]
    save("M7_U1_Observabilidad.pdf",obs)
    demo=json.loads((ROOT/"notebooks/preentrega6_demo.ipynb").read_text())
    supervisor=from_markdown((ROOT/"docs/preentrega6.md").read_text())
    supervisor.insert(1,paragraph("Indice: objetivo, arquitectura, herramientas, estado compartido, validacion, demostracion ejecutada y limitaciones."))
    supervisor += [Paragraph("Demostracion ejecutada",h1)]
    for cell in demo["cells"]:
        if cell["cell_type"]=="code":
            for output in cell.get("outputs",[]):
                if output.get("output_type")=="stream":
                    text="".join(output.get("text",[]))
                    if text.startswith("{'answer'"):
                        supervisor.append(paragraph("El notebook conserva la respuesta completa y sus referencias. Se verifico validation_passed=True y contribuciones de researcher y analyst."))
                    else: supervisor.append(paragraph(text))
    supervisor += [Paragraph("Limites",h1),paragraph("El corpus es educativo y ficticio. La validacion demuestra pertenencia de referencias y contrato, no verdad universal. El notebook ejecutado es parte del repositorio y no contiene credenciales. El sistema no se publica como servicio de Internet.")]
    save("M6_Preentrega6_Orquestador.pdf",supervisor)
    lcel=json.loads((ROOT/"notebooks/m2u1_lcel_demo.ipynb").read_text())
    text="".join(lcel["cells"][1]["outputs"][0]["text"])
    story=[Paragraph("Modulo 2, unidad 1 - LCEL asincrono",h1),
           paragraph("Mauro Tomas Proto Cassina. 26 de septiembre de 2026."),
           Paragraph("Indice",h1),paragraph("Objetivo, composicion de la cadena, seguridad, codigo y resultado real."),
           Paragraph("Objetivo y composicion",h1),paragraph("La pregunta entra como un diccionario. ChatPromptTemplate construye los roles system y human. ChatOpenAI funciona como adaptador compatible con DeepSeek. StrOutputParser extrae el texto de la respuesta. Los componentes se conectan con el operador | y se ejecutan con await chain.ainvoke. No se usa una clave de OpenAI."),
           Paragraph("Seguridad",h1),paragraph("La credencial DeepSeek llega solo al entorno del proceso y no se muestra ni se guarda. El destino HTTPS esta fijo; los clientes ignoran proxies y redirecciones. No se descargan modelos."),
           Paragraph("Resultado real",h1),paragraph(text),PageBreak(),Paragraph("Codigo entregado",h1),
           Preformatted((ROOT/"exercises/m2u1_lcel.py").read_text(),code,maxLineLength=92)]
    save("M2_U1_LCEL.pdf",story)
    html="<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Informe de observabilidad - AI Engineering</title><style>body{max-width:760px;margin:42px auto;padding:0 22px;font:18px/1.6 Georgia,serif;color:#202020;background:#fff}h1,h2{line-height:1.25}a{color:#184e8e}</style><main>"
    for block in evaluation.split("\n\n"):
        text=" ".join(block.strip().splitlines())
        if text.startswith("## "): html+="<h2>"+escape(text[3:])+"</h2>"
        elif text.startswith("# "): html+="<h1>"+escape(text[2:])+"</h1>"
        elif text: html+="<p>"+escape(text)+"</p>"
    html+="<h2>Resultados por consulta</h2>"
    for index,j in enumerate(jobs,1):
        html+=f"<h3>Consulta {index}</h3><p>{escape(j['request']['query'])}</p><p>{escape(j['result']['answer'])}</p><p>Latencia {j['latency_seconds']:.6f} s. Costo estimado {j['cost_usd']:.9f} USD. Trace ID {j['trace_id']}.</p>"
    html+="<p><a href='https://github.com/MauroProto/ai-engineering-intelligence'>Repositorio y evidencia original</a> · <a href='M7_U1_Observabilidad.pdf'>PDF del informe</a></p><h2>Capturas originales de Phoenix</h2>"
    for name in ("phoenix-cinco-ejecuciones.png","phoenix-p95.png","phoenix-traza-nodos.png"):
        html+=f"<figure><img style='width:100%' src='https://raw.githubusercontent.com/MauroProto/ai-engineering-intelligence/main/screenshots/{name}' alt='{name}'><figcaption>{name}</figcaption></figure>"
    html+="</main></html>"
    (ROOT/"docs/index.html").write_text(html,encoding="utf-8")
    (ROOT/"docs/M7_U1_Observabilidad.pdf").write_bytes((OUT/"M7_U1_Observabilidad.pdf").read_bytes())
    print("Tres PDF sencillos e informe HTML generados")


if __name__ == "__main__": main()
