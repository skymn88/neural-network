"""Editable, template-aware overview; reads static CONTENT/NUMBERS only, never trains."""

import argparse
import uuid
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.chart.data import XyChartData
from pptx.oxml.xmlchemy import OxmlElement

# SETTINGS — edit these or pass --template/--output. None preserves template fonts.
SETTINGS = dict(
    template=None,
    font="Franklin Gothic Medium",
    output=str(
        Path(__file__).resolve().parents[1]
        / "presentations/actuator_black_box_overview.pptx"
    ),
)
# THEME — text is black; backgrounds white. Diagram lines may use muted gray.
THEME = dict(
    background="FFFFFF",
    text="000000",
    line="707070",
    measured="000000",
    predicted="888888",
)
# NUMBERS — all illustrative numerical values are centralized and editable.
NUMBERS = dict(
    sample_rate=64,
    interface_rate=200,
    split="7 / 2 / 2",
    time=[0, 1, 2, 3, 4, 5, 6, 7],
    hold_measured=[0, 0, -30, -30, -30, -30, -30, -30],
    hold_predicted=[0, 0, -27, -25, -23, -21, -19, -17],
    step_measured=[0, 0, 0, -30, -30, -30, -30, -30],
    step_predicted=[0, 0, 0, -8, -17, -24, -28, -30],
    vibration_measured=[0, 0, -20, 20, -20, 20, -20, 0],
    vibration_predicted=[0, 0, -8, 8, -8, 8, -8, 0],
    offset_measured=[0, 0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    offset_predicted=[0, 0, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02],
)
# CONTENT — presentation wording is general, not rewritten from the latest experiment.
CONTENT_TEXT = dict(
    legend=("Ölçülen davranış — şematik", "Tahmin davranışı — şematik"),
    notes="Genel proje anlatımı. Şematik grafikler NUMBERS bloğundan gelir; gerçek uçuş veya yeni deney sonucu değildir.",
)
CONTENT = [
    dict(
        title="Horizontal tail aktüatörü: veriden modelleme",
        bullets=[
            "Amaç, elektrik akımından yüzey hareketini tahmin etmektir.",
            "Model, uçuş kayıtlarındaki zaman ilişkilerini öğrenir.",
            "Başarı, modelin görmediği uçuşlarla sınanır.",
        ],
        visual="chain",
        labels=[
            "Akım",
            "DDV (Direct Drive Valve)",
            "RAM (aktüatör pistonu)",
            "Yüzey açısı",
        ],
    ),
    dict(
        title="Ne öğrenmek istiyoruz?",
        bullets=[
            "Akım, valfin konumunu etkiler.",
            "Valf hareketi, pistonun hızını ve konumunu etkiler.",
            "Piston konumu, yüzey açısına dönüşür.",
            "Aynı girişin etkisi önceki duruma bağlı olabilir.",
        ],
        visual="chain",
        labels=["Giriş geçmişi", "İç durum", "Tahmin"],
    ),
    dict(
        title="Veri hazırlama",
        bullets=[
            "Her sinyal kendi zaman bilgisiyle okunur.",
            "Eksik örnekler ve zaman kaymaları kontrol edilir.",
            "Yüksek örneklemeli veri süzülerek ortak zaman tabanına getirilir.",
            "Birleştirme ve ölçekler yalnız eğitim verisiyle belirlenir.",
        ],
        visual="chain",
        labels=["CSV kayıtları", "Zaman hizalama", "Ortak örnekleme"],
    ),
    dict(
        title="Cascade: alt modellerin zinciri",
        bullets=[
            "DDV (Direct Drive Valve), valf sürgüsü konumunu ifade eder.",
            "RAM burada aktüatör pistonu veya koludur; bilgisayar belleği değildir.",
            "Her alt modelin hatası sonraki aşamaya taşınabilir.",
            "Zincirin son çıktısı, uygulamadaki asıl başarı ölçüsüdür.",
        ],
        visual="chain",
        labels=["Akım → DDV", "DDV → RAM", "RAM → Açı"],
    ),
    dict(
        title="Component ve Cascade farkı",
        bullets=[
            "Component değerlendirmesi, alt modele ölçülen ara sinyali verir.",
            "Cascade değerlendirmesi, önceki modelin tahminini verir.",
            "İki sonuç arasındaki fark, hatanın nerede büyüdüğünü gösterir.",
            "Bu iki skoru ortalamak sistem doğruluğu anlamına gelmez.",
        ],
        visual="chain",
        labels=["Ölçülen DDV → RAM", "Tahmin DDV → RAM"],
    ),
    dict(
        title="Eğitim, validation ve test",
        bullets=[
            "Eğitim, model parametrelerini öğrenir.",
            "Validation, yöntem ve ayar seçimine yardım eder.",
            "Test, seçim bittikten sonra bağımsız değerlendirme içindir.",
            "Tek uçuş içindeki zaman bölmesi, yeni uçuş doğrulaması değildir.",
        ],
        visual="chain",
        labels=["TRAIN: öğrenme", "VALIDATION: seçim", "TEST: son kontrol"],
    ),
    dict(
        title="Hafıza neden gerekli?",
        bullets=[
            "Kısa bir akım darbesi yüzeyi yeni bir konuma taşıyabilir.",
            "Giriş tekrar sıfıra geldiğinde konum korunabilir.",
            "DMLP (Dynamic Multi-Layer Perceptron), geçmiş veya durum girdileri kullanır.",
            "Bir-adım başarısı, uzun serbest koşunun doğru olacağını garanti etmez.",
        ],
        visual="step",
    ),
    dict(
        title="Yöntemler: güçlü ve zayıf taraflar",
        bullets=[
            "Basit yöntemler iyi bir başlangıç ve karşılaştırma sağlar.",
            "Daha karmaşık ağ, her veri kümesinde daha iyi değildir.",
        ],
        visual="methods",
        rows=[
            ["Yöntem", "Güçlü taraf", "Zayıf taraf"],
            [
                "FIR (Finite Impulse Response)",
                "Kolay çıkarım",
                "Uzun hafıza için çok gecikme",
            ],
            [
                "OE (Output Error)",
                "Konum hatasına doğrudan bakar",
                "Uzun koşuda hata birikebilir",
            ],
            [
                "NL_FB (Nonlinear Filter Bank)",
                "Farklı zaman ölçekleri",
                "Baz seçimine duyarlı",
            ],
            [
                "MLP (Multi-Layer Perceptron)",
                "Esnek statik harita",
                "Hafıza ayrıca gerekir",
            ],
            ["LSTM / GRU", "Öğrenilmiş zaman hafızası", "Eğitim ve veri gereksinimi"],
            [
                "ESN (Echo State Network)",
                "Hızlı çıkış katmanı eğitimi",
                "Rezervuar seçimine duyarlı",
            ],
        ],
    ),
    dict(
        title="Rekürrent ağlar",
        bullets=[
            "LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit), geçmiş bilgiyi iç durumda tutar.",
            "NARX (Nonlinear AutoRegressive with Exogenous Inputs), geçmiş giriş ve çıktıları kullanır.",
            "Ölçülen çıktı ile eğitilen ağ, kendi tahminleriyle çalışırken sapabilir.",
            "Uzun serbest koşu mutlaka ayrıca değerlendirilir.",
        ],
        visual="chain",
        labels=["Giriş", "Hafıza durumu", "Hız düzeltmesi"],
    ),
    dict(
        title="Bulgular: tutma bölgesi düz değil",
        bullets=[
            "Ölçülen yüzey sabitken tahmin yavaşça kayabilir.",
            "Küçük bir hız veya null hatası zamanla konum hatasına dönüşür.",
            "Tutma sürüklenmesi, birim/s olarak ayrıca raporlanır.",
            "Sağdaki çizim, uçuş figürlerinde bildirilen davranışın şemasıdır.",
        ],
        visual="hold",
    ),
    dict(
        title="Bulgular: geçişler yuvarlak",
        bullets=[
            "Gerçek hareket kısa sürede tamamlanırken tahmin uzun bir rampa yapabilir.",
            "Aşırı sönüm, hız sınırı veya yetersiz hafıza bu davranışı etkileyebilir.",
            "Geçiş süresi ve tepe hız, Fit yanında kontrol edilir.",
            "Çizim şematiktir; yeni bir uçuş ölçümü değildir.",
        ],
        visual="step",
    ),
    dict(
        title="Bulgular: titreşim genliği küçük",
        bullets=[
            "Tahmin doğru yönde salınsa da genliği düşük kalabilir.",
            "Genlik hatası ile zaman kayması ayrı incelenir.",
            "Frekans bantlarında genlik oranı ve korelasyon karşılaştırılır.",
            "Çizim, bildirilen davranışı açıklayan şemadır.",
        ],
        visual="vibration",
    ),
    dict(
        title="Bulgular: DDV seviyesi kayık",
        bullets=[
            "DDV (Direct Drive Valve) tahmininde küçük bir seviye farkı görülebilir.",
            "Bu fark RAM hız haritasına yanlış bir giriş verebilir.",
            "Null seviyesi uçuş boyunca değişebilir.",
            "Gözlenmeyen yük veya ofset, yalnız akımdan kesin belirlenemeyebilir.",
        ],
        visual="offset",
    ),
    dict(
        title="Genelleme ve güvenilir seçim",
        bullets=[
            "Aynı ayar farklı eğitim seed değerleriyle sınanır.",
            "Ensemble, birden fazla modelin hız çıktısını ortalar.",
            "Uçuşların ortalama ve en düşük Fit değerleri birlikte gösterilir.",
            "Sınırlı sayıda uçuşta, uçuş dışarıda bırakma değerlendirmesi kullanılır.",
        ],
        visual="ensemble",
        labels=["Model A", "Model B", "Ortalama hız → RAM"],
    ),
    dict(
        title="Gelecek çalışmalar",
        bullets=[
            "Gerçek uçuşlarda birim, hizalama ve DDV ofsetlerini doğrulamak.",
            "Giriş kapsamını farklı hareket ve yük koşullarıyla genişletmek.",
            "Hata ve belirsizliği uçuş bazında raporlamak.",
            "Sabit durumlu çıkarımı C ve Simulink S-Function arayüzüne taşımak.",
        ],
        visual="chain",
        labels=["Gerçek uçuş kontrolü", "Bağımsız doğrulama", "Uygulamaya aktarım"],
    ),
    dict(
        title="Kısaltmalar sözlüğü",
        bullets=[],
        visual="glossary",
        rows=[
            ["Kısaltma", "Uzun hali / anlamı"],
            ["DDV", "Direct Drive Valve — doğrudan sürülen valf"],
            ["RAM", "Aktüatör pistonu/kolu; burada bellek kısaltması değildir"],
            ["NN / MLP", "Neural Network / Multi-Layer Perceptron"],
            ["DMLP", "Dynamic Multi-Layer Perceptron"],
            ["FIR", "Finite Impulse Response"],
            ["OE", "Output Error"],
            ["NL_FB", "Nonlinear Filter Bank"],
            ["LSTM", "Long Short-Term Memory"],
            ["GRU", "Gated Recurrent Unit"],
            ["NARX", "Nonlinear AutoRegressive with Exogenous Inputs"],
            ["ESN", "Echo State Network"],
            ["TCN", "Temporal Convolutional Network"],
            ["RBF", "Radial Basis Function"],
            ["S-Function", "Simulink sistem fonksiyonu arayüzü"],
        ],
    ),
]


def theme_font(path, font):
    if font is None:
        return
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    with ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    for name, data in list(parts.items()):
        if name.startswith("ppt/theme/") and name.endswith(".xml"):
            root = etree.fromstring(data)
            for node in root.xpath(
                ".//a:fontScheme/a:majorFont/a:latin | .//a:fontScheme/a:minorFont/a:latin",
                namespaces=ns,
            ):
                node.set("typeface", font)
            parts[name] = etree.tostring(
                root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)


def build(template=None, output=None, font=...):
    if font is Ellipsis:
        font = SETTINGS["font"]
    prs = (
        Presentation(template or SETTINGS["template"])
        if (template or SETTINGS["template"])
        else Presentation()
    )
    if not (template or SETTINGS["template"]):
        prs.slide_width = 12192000
        prs.slide_height = 6858000
    for sid in list(prs.slides._sldIdLst):
        prs.part.drop_rel(sid.rId)
        prs.slides._sldIdLst.remove(sid)
    W, H = prs.slide_width, prs.slide_height
    layout = next(
        (
            l
            for l in prs.slide_layouts
            if any(
                p.placeholder_format.type
                in (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)
                for p in l.placeholders
            )
        ),
        None,
    )
    if layout is None:
        raise ValueError("Template needs a layout with a real title placeholder")

    def rect(x, y, w, h):
        return tuple(int(v * s) for v, s in zip((x, y, w, h), (W, H, W, H)))

    def fmt(run, size=20):
        run.font.size = Pt(size)
        run.font.bold = False
        run.font.color.rgb = RGBColor.from_string(THEME["text"])
        if font is not None:
            run.font.name = font

    def number(sl, n):
        sh = sl.shapes.add_textbox(*rect(0.91, 0.94, 0.05, 0.04))
        p = sh.text_frame.paragraphs[0]
        field = OxmlElement("a:fld")
        field.set("id", "{" + str(uuid.uuid4()) + "}")
        field.set("type", "slidenum")
        rp = OxmlElement("a:rPr")
        rp.set("sz", "1200")
        fill = OxmlElement("a:solidFill")
        color = OxmlElement("a:srgbClr")
        color.set("val", THEME["text"])
        fill.append(color)
        rp.append(fill)
        if font is not None:
            latin = OxmlElement("a:latin")
            latin.set("typeface", font)
            rp.append(latin)
        field.append(rp)
        text = OxmlElement("a:t")
        text.text = str(n)
        field.append(text)
        p._p.append(field)

    def chart(sl, kind):
        pairs = {
            "hold": ("hold_measured", "hold_predicted"),
            "step": ("step_measured", "step_predicted"),
            "vibration": ("vibration_measured", "vibration_predicted"),
            "offset": ("offset_measured", "offset_predicted"),
        }
        data = XyChartData()
        for label, key in zip(CONTENT_TEXT["legend"], pairs[kind]):
            series = data.add_series(label)
            for x, y in zip(NUMBERS["time"], NUMBERS[key]):
                series.add_data_point(x, y)
        c = sl.shapes.add_chart(
            XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
            *rect(0.51, 0.25, 0.45, 0.58),
            data,
        ).chart
        c.has_legend = True
        c.legend.position = XL_LEGEND_POSITION.BOTTOM
        c.legend.font.size = Pt(11)
        c.legend.font.color.rgb = RGBColor.from_string(THEME["text"])
        if font is not None:
            c.legend.font.name = font
        for axis in (c.category_axis, c.value_axis):
            axis.tick_labels.font.size = Pt(11)
            axis.tick_labels.font.color.rgb = RGBColor.from_string(THEME["text"])
            if font is not None:
                axis.tick_labels.font.name = font
        for series, color in zip(c.series, (THEME["measured"], THEME["predicted"])):
            series.format.line.color.rgb = RGBColor.from_string(color)
            series.format.line.width = Pt(2)

    for index, item in enumerate(CONTENT, 1):
        sl = prs.slides.add_slide(layout)
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = RGBColor.from_string(THEME["background"])
        title = sl.shapes.title
        for sh in list(sl.shapes):
            if sh.is_placeholder and sh != title:
                sh._element.getparent().remove(sh._element)
        title.left, title.top, title.width, title.height = rect(0.04, 0.04, 0.92, 0.12)
        title.text = item["title"]
        for p in title.text_frame.paragraphs:
            for r in p.runs:
                fmt(r, 28)
        if item["bullets"]:
            body = sl.shapes.add_textbox(*rect(0.04, 0.23, 0.43, 0.64))
            body.name = "content_bullets"
            tf = body.text_frame
            tf.word_wrap = True
            for j, line in enumerate(item["bullets"]):
                p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                p.text = line
                p.space_after = Pt(16)
                pp = p._p.get_or_add_pPr()
                pp.set("marL", "228600")
                pp.set("indent", "-171450")
                bullet = OxmlElement("a:buChar")
                bullet.set("char", "•")
                pp.append(bullet)
                for r in p.runs:
                    fmt(r, 19 if item["visual"] == "methods" else 21)
        visual = item["visual"]
        if visual in ("hold", "step", "vibration", "offset"):
            chart(sl, visual)
        elif visual == "chain":
            labels = item["labels"]
            height = min(0.14, 0.56 / len(labels))
            top = 0.24
            for j, label in enumerate(labels):
                node = sl.shapes.add_shape(
                    MSO_SHAPE.ROUNDED_RECTANGLE,
                    *rect(0.56, top + j * (height + 0.035), 0.36, height),
                )
                node.fill.solid()
                node.fill.fore_color.rgb = RGBColor.from_string(THEME["background"])
                node.line.color.rgb = RGBColor.from_string(THEME["line"])
                node.text = label
                for p in node.text_frame.paragraphs:
                    for r in p.runs:
                        fmt(r, 18)
                if j < len(labels) - 1:
                    arrow = sl.shapes.add_shape(
                        MSO_SHAPE.DOWN_ARROW,
                        *rect(
                            0.725,
                            top + j * (height + 0.035) + height + 0.005,
                            0.03,
                            0.025,
                        ),
                    )
                    arrow.fill.solid()
                    arrow.fill.fore_color.rgb = RGBColor.from_string(THEME["line"])
                    arrow.line.fill.background()
        elif visual == "ensemble":
            for label, area in zip(
                item["labels"],
                [
                    (0.51, 0.28, 0.20, 0.13),
                    (0.76, 0.28, 0.20, 0.13),
                    (0.57, 0.59, 0.34, 0.15),
                ],
            ):
                node = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, *rect(*area))
                node.text = label
                node.fill.solid()
                node.fill.fore_color.rgb = RGBColor.from_string(THEME["background"])
                node.line.color.rgb = RGBColor.from_string(THEME["line"])
                for p in node.text_frame.paragraphs:
                    for r in p.runs:
                        fmt(r, 18)
            for left in (0.61, 0.83):
                arrow = sl.shapes.add_shape(
                    MSO_SHAPE.DOWN_ARROW, *rect(left, 0.44, 0.035, 0.12)
                )
                arrow.fill.solid()
                arrow.fill.fore_color.rgb = RGBColor.from_string(THEME["line"])
                arrow.line.fill.background()
        elif visual in ("methods", "glossary"):
            rows = item["rows"]
            area = (
                (0.49, 0.23, 0.47, 0.63)
                if visual == "methods"
                else (0.04, 0.21, 0.92, 0.69)
            )
            table = sl.shapes.add_table(len(rows), len(rows[0]), *rect(*area)).table
            if visual == "glossary":
                table.columns[0].width = int(W * 0.19)
                table.columns[1].width = int(W * 0.73)
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    cell = table.cell(i, j)
                    cell.text = value
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor.from_string(THEME["background"])
                    for p in cell.text_frame.paragraphs:
                        for r in p.runs:
                            fmt(r, 13 if visual == "methods" else 15)
        number(sl, index)
        sl.notes_slide.notes_text_frame.text = CONTENT_TEXT["notes"]
    output = Path(output or SETTINGS["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)
    theme_font(output, font)
    print(str(output))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=SETTINGS["template"])
    parser.add_argument("--output", default=SETTINGS["output"])
    args = parser.parse_args()
    build(args.template, args.output)
