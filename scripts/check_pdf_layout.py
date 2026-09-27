"""Hoja de control para inspeccion visual de todas las paginas renderizadas."""
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
root=Path(__file__).resolve().parents[1]
files=sorted((root/"tmp/pdfs").glob("*/page-*.png"))
for start in range(0,len(files),6):
    sheet=Image.new("RGB",(1300,1860),"#ddd")
    draw=ImageDraw.Draw(sheet)
    for index,path in enumerate(files[start:start+6]):
        im=Image.open(path).convert("RGB")
        im.thumbnail((620,860))
        x=20+(index%2)*650;y=30+(index//2)*620
        im.thumbnail((430,570))
        sheet.paste(im,(x,y))
        draw.text((x,y-20),str(path.relative_to(root/"tmp/pdfs")),fill="black")
    target=root/"tmp/pdfs"/f"control-{start//6+1}.png"
    sheet.save(target)
    print(target)
