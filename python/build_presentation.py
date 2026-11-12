"""Build an editable PowerPoint from recorded results; never retrains models.
Requires only the optional requirements-presentations.txt plus app NumPy/SciPy.
All text, diagrams, tables and charts are native editable PowerPoint objects.
"""

import json
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_MARK
from pptx.chart.data import XyChartData
from actuator.data import ROOT, pulse_flights
from actuator.inference import rollout

THEME = {
    "bg": "0E1925",
    "card": "192B3C",
    "teal": "55D9C6",
    "blue": "61A8F5",
    "amber": "F3BE69",
    "ink": "F4F7FA",
    "muted": "ADC0D0",
    "dark": "132D40",
    "white": "FFFFFF",
}
NEW = "20261112_105714_655119"
PREVIOUS = "20261112_062616_313440"
OUT = ROOT / "presentations/actuator_black_box_overview.pptx"


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "Horizontal Tail Actuator — Black-Box Modelleme"
    prs.core_properties.subject = (
        "MATLAB geçmişi, Python deneyleri, Python 3.9 çevrimdışı kullanım"
    )
    prs.core_properties.author = "Actuator Project"
    records = {}
    for name in [NEW, PREVIOUS, "20261112_053325_290124", "20261112_053356_805234"]:
        p = ROOT / "python/reference_results" / name
        records[name] = json.loads((p / "result.json").read_text())
    result = records[NEW]
    selected = result["selected"]
    models = json.loads(
        (ROOT / "python/reference_results" / NEW / "models.json").read_text()
    )
    names = json.loads(
        (ROOT / "python/reference_results" / PREVIOUS / "flights.json").read_text()
    )
    target = next(r["flight"] for r in result["split"] if r["split"] == "validation")
    ix = names.index(target)
    raw = ROOT / "python/reference_results" / PREVIOUS / "flights.npz"
    if raw.exists():
        with np.load(raw, allow_pickle=False) as a:
            flight = a["f" + str(ix)].copy()
    else:
        fs, _ = pulse_flights(seed=621, profile="flightLike")
        f = fs[ix]
        flight = np.c_[f.time, f.values]
    new = rollout(models[selected], flight[:, 1], flight[0, 3])

    def color(key):
        return RGBColor.from_string(THEME.get(key, key))

    def box(sl, x, y, w, h, fill="card", radius=True, line=None):
        sh = sl.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
            Inches(x),
            Inches(y),
            Inches(w),
            Inches(h),
        )
        sh.fill.solid()
        sh.fill.fore_color.rgb = color(fill)
        if line:
            sh.line.color.rgb = color(line)
        else:
            sh.line.fill.background()
        if radius:
            try:
                sh.adjustments[0] = 0.12
            except (IndexError, AttributeError):
                pass
        return sh

    def text(sl, x, y, w, h, value, size=20, fill="ink", bold=False):
        sh = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = 0
        tf.margin_top = tf.margin_bottom = 0
        for i, line in enumerate(str(value).split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = "Aptos"
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = color(fill)
            p.space_after = Pt(0)
        return sh

    def slide(kicker, title, sub=""):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = color("bg")
        box(sl, 0.55, 0.38, 0.12, 0.32, "teal", False)
        text(sl, 0.8, 0.39, 11, 0.25, kicker.upper(), 11, "teal", True)
        text(sl, 0.65, 0.95, 12.05, 0.85, title, 30, bold=True)
        if sub:
            text(sl, 0.68, 1.85, 11.9, 0.55, sub, 14, "muted")
        text(
            sl,
            0.65,
            7.08,
            11,
            0.22,
            "ACTUATOR LAB  /  12 KASIM 2026  /  SENTETİK ≠ GERÇEK UÇUŞ DOĞRULAMASI",
            9,
            "muted",
        )
        text(sl, 12, 7.03, 0.7, 0.3, f"{len(prs.slides):02d}", 12, "teal", True)
        return sl

    def card(sl, x, y, w, h, title, body, accent="teal"):
        box(sl, x, y, w, h)
        text(sl, x + 0.22, y + 0.2, w - 0.44, 0.6, title, 20, accent, True)
        text(sl, x + 0.22, y + 0.95, w - 0.44, h - 1.08, body, 16, "muted")

    def table(sl, x, y, width, headers, rows, widths=None):
        rh = 0.55
        shape = sl.shapes.add_table(
            len(rows) + 1,
            len(headers),
            Inches(x),
            Inches(y),
            Inches(width),
            Inches(rh * (len(rows) + 1)),
        )
        tb = shape.table
        if widths:
            for col, w in zip(tb.columns, widths):
                col.width = Inches(w)
        for ri, row in enumerate([headers] + rows):
            for ci, value in enumerate(row):
                c = tb.cell(ri, ci)
                c.fill.solid()
                c.fill.fore_color.rgb = color("dark" if ri == 0 else "card")
                c.margin_left = Inches(0.12)
                c.margin_top = Inches(0.08)
                c.text = str(value)
                for p in c.text_frame.paragraphs:
                    p.font.name = "Aptos"
                    p.font.size = Pt(14 if ri else 13)
                    p.font.bold = ri == 0
                    p.font.color.rgb = color("teal" if ri == 0 else "ink")
        return shape

    def metric(sl, x, y, w, label, value, detail="", accent="teal"):
        box(sl, x, y, w, 1.55)
        text(sl, x + 0.2, y + 0.15, w - 0.4, 0.3, label, 13, "muted")
        text(sl, x + 0.2, y + 0.5, w - 0.4, 0.65, value, 33, accent, True)
        text(sl, x + 0.2, y + 1.2, w - 0.4, 0.23, detail, 11, "muted")

    def chart(sl, x, y, w, h, series, title, limits=None):
        box(sl, x, y, w, h, "white")
        data = XyChartData()
        for name, t, values, col in series:
            s = data.add_series(name)
            # Shared uniform-time extrema selection keeps peaks without huge slide workbooks.
            ids = {0, len(t) - 1}
            for group in np.array_split(np.arange(len(t)), min(180, len(t))):
                ids.add(int(group[np.argmin(values[group])]))
                ids.add(int(group[np.argmax(values[group])]))
            for i in sorted(ids):
                s.add_data_point(float(t[i]), float(values[i]))
        ch = sl.shapes.add_chart(
            XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
            Inches(x + 0.08),
            Inches(y + 0.08),
            Inches(w - 0.16),
            Inches(h - 0.16),
            data,
        ).chart
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        for p in ch.chart_title.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.name = "Aptos"
            p.font.color.rgb = color("dark")
        ch.has_legend = True
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(9)
        for se, entry in zip(ch.series, series):
            se.format.line.color.rgb = color(entry[3])
            se.format.line.width = Pt(1.2)
        for ax in [ch.category_axis, ch.value_axis]:
            ax.tick_labels.font.size = Pt(9)
            ax.tick_labels.font.color.rgb = color("dark")
            ax.major_tick_mark = XL_TICK_MARK.NONE
        ch.category_axis.tick_labels.number_format = "0"
        ch.value_axis.tick_labels.number_format = "0.0"
        if limits:
            ch.value_axis.minimum_scale = limits[0]
            ch.value_axis.maximum_scale = limits[1]
        return ch

    def fits(rec, stage="RAM_Cascade", model=None):
        row = next(
            r
            for r in rec["summary"]
            if r["model"] == (model or rec["selected"]) and r["stage"] == stage
        )
        return [f"{row['fit_' + s]:.2f}" for s in ["train", "validation", "test"]]

    # 01
    sl = slide(
        "Proje özeti",
        "Akımdan harekete.",
        "Horizontal tail aktüatörünün veriye dayalı, black-box dinamik modeli",
    )
    text(
        sl,
        0.75,
        2.65,
        6.5,
        1.5,
        "Öğrenilmiş dinamikler.\nÖlçülebilir ilerleme.",
        34,
        bold=True,
    )
    text(
        sl,
        0.75,
        4.35,
        6.3,
        1.15,
        "MATLAB deneylerinden Python 3.9’a:\nmodeller, doğrulama ve çevrimdışı kullanım.",
        21,
        "muted",
    )
    for i, (label, caption) in enumerate(
        [
            ("CURRENT", "Ölçülen giriş"),
            ("DDV", "Valf hareketi"),
            ("RAM", "Durum hafızası"),
            ("ANGLE", "Geometrik dönüşüm"),
        ]
    ):
        yy = 2.45 + i * 0.9
        box(sl, 8, yy, 4.35, 0.72, "card")
        text(sl, 8.25, yy + 0.1, 1.5, 0.3, label, 18, "teal", True)
        text(sl, 9.8, yy + 0.15, 2.2, 0.3, caption, 12, "muted")
    # 02
    sl = slide(
        "01 / Hedef",
        "Fiziksel denklemi değil, davranışı öğreniyoruz.",
        "İleri kullanım hedefi: input ver, fiziksel birimde output al; daha sonra C / S-Function.",
    )
    for i, (a, b) in enumerate(
        [
            ("Akım → DDV", "Nedensel giriş geçmişi\nGürültüyü azaltan dinamik eşleme"),
            ("DDV → RAM", "Hız + konum hafızası\nDarbe, tutma, doygunluk"),
            ("RAM → Açı", "Ölçümden öğrenilmiş\nstatik monoton ilişki"),
        ]
    ):
        card(sl, 0.7 + i * 4.2, 2.7, 3.95, 2.65, a, b)
    text(
        sl,
        0.8,
        5.8,
        11.8,
        0.65,
        "Serbest koşu: yalnız akım + başlangıç RAM. Gerçek DDV/RAM ile sürekli düzeltme yok.",
        20,
        "teal",
    )
    # 03
    sl = slide(
        "02 / Veri sözleşmesi",
        "Karışık örnekleme → ortak 64 Hz ekseni.",
        "Sinyaller kendi zaman vektörlerini korur; hizalama ve birim tutarlılığı önce gelir.",
    )
    table(
        sl,
        0.75,
        2.55,
        11.8,
        ["Sinyal", "Bilinen örnekleme", "Bu çalışma"],
        [
            ["Coil Current / DDV", "64 Hz", "Aktif giriş / ara çıktı"],
            [
                "RAM position / Deflection",
                "64 Hz; RAM ayrıca 1000 Hz",
                "64 Hz kullanılıyor",
            ],
            ["Desired komutlar", "1 Hz", "Kullanılmıyor"],
            [
                "Supply / return pressure",
                "2048 / 128 Hz",
                "Bu Python zincirinde kullanılmıyor",
            ],
        ],
        [4.25, 3.25, 4.3],
    )
    text(
        sl,
        0.85,
        5.75,
        11.6,
        0.65,
        "Uçuşlar arasında birim, işaret ve sensör sıfırı değişirse ağ bunu fiziksel dinamik sanabilir.",
        19,
        "amber",
    )
    # 04
    sl = slide(
        "03 / Süreç",
        "Yöntemleri sonuçlara göre daralttık.",
        "Tüm denemeler ve başarısız sonuçlar changelog’da tutuluyor.",
    )
    phases = [
        (
            "MATLAB",
            "MLP / NARX → FIR / NL_OE\nNL_FB → NLSS / DMLP",
            "Gerçek veri sonuçları kullanıcı tarafından bildirildi.",
        ),
        (
            "PYTHON",
            "OE / dinamik NN\nOrtak DDV–RAM eğitimi",
            "Bu ortamda gerçek eğitim ve test koşuları çalıştırıldı.",
        ),
        (
            "FLIGHTLIKE",
            "Null / hız haritası\nSakinlik + osilasyon izleyici",
            "Daha zor sentetik profil ve sınırlı iteratif arama.",
        ),
    ]
    for i, (a, b, c) in enumerate(phases):
        xx = 0.75 + i * 4.2
        card(sl, xx, 2.65, 3.95, 3.25, a, b)
        text(sl, xx + 0.22, 5, 3.5, 0.6, c, 12, "muted")
    text(
        sl,
        0.85,
        6.25,
        11.7,
        0.3,
        "MATLAB kaynakları dev_Codex dalında; bu dal artık Python odaklı.",
        16,
        "teal",
    )
    # 05
    sl = slide(
        "04 / Ölçüm",
        "Tek adım başarısı, serbest koşu başarısı değildir.",
        "Model seçimi yalnız validation RAM Cascade RMSE’ye dayanır.",
    )
    metric(
        sl, 0.75, 2.6, 3.9, "11 uçuş / split", "7 · 2 · 2", "TRAIN / VALIDATION / TEST"
    )
    metric(sl, 4.95, 2.6, 3.9, "Hedef", "≥ %95", "Validation RAM Cascade Fit")
    metric(sl, 9.15, 2.6, 3.45, "İdeal skor", "100 / 0", "Fit / RMSE")
    card(
        sl,
        0.75,
        4.55,
        5.75,
        1.95,
        "Component",
        "Ölçülen ara girdiyle alt modeli değerlendirir. Hata kaynağını ayırır.",
    )
    card(
        sl,
        6.8,
        4.55,
        5.75,
        1.95,
        "Cascade",
        "Tahmin edilen ara girdiyi kullanır. Uçtan uca fidelity burada ölçülür.",
    )
    # 06
    sl = slide(
        "05 / Yöntem haritası",
        "Her yöntem aynı sorunu çözmedi.",
        "Kısa karar özeti; farklı veri/protokol sonuçları doğrudan sıralanamaz.",
    )
    table(
        sl,
        0.7,
        2.45,
        11.95,
        ["Aile", "Ne öğreniyor?", "Karar / gözlem"],
        [
            [
                "MLP / dinamik MLP",
                "Statik veya durumlu doğrusal olmayan eşleme",
                "Python’da güçlü sentetik referans",
            ],
            [
                "NARX",
                "Çıkış geçmişiyle otoregresif dinamik",
                "Serbest koşuda ciddi sapma; aktif değil",
            ],
            [
                "FIR / NL_OE / NL_FB",
                "Giriş geçmişi veya filtre bankasından hız/konum",
                "NL_FB: gerçek veri raporlarında güçlü",
            ],
            [
                "NLSS / WH / GP",
                "Durum, statik nonlineerlik veya kernel",
                "Kazanç veri/ayar bağımlı; bazıları pahalı",
            ],
            [
                "Null / hız haritası",
                "Öğrenilmiş hız + sızıntısız integral",
                "Yeni flightLike profilinde en güçlü aday",
            ],
        ],
        [3.4, 4.25, 4.3],
    )
    # 07
    sl = slide(
        "06 / Kullanıcı tarafından bildirilen",
        "Gerçek uçuşlarda NL_FB öne çıktı.",
        "Bu tablolar burada yeniden çalıştırılmadı; geçmiş kullanıcı raporlarıdır.",
    )
    table(
        sl,
        0.75,
        2.6,
        11.8,
        ["Yöntem / aşama", "Train Fit", "Val Fit", "Test Fit"],
        [
            ["NL_OE / RAM Cascade", "≈70", "≈61", "≈39"],
            ["NL_FB / RAM Cascade", "≈73", "≈65", "≈40"],
            ["NLSS / RAM Component", "≈63", "≈18", "≈38"],
        ],
        [5.5, 2.1, 2.1, 2.1],
    )
    card(
        sl,
        0.75,
        5,
        5.75,
        1.65,
        "En önemli sorunlar",
        "Null/ofset, basamak sonrası kuyruk, eksik genlik.",
        accent="amber",
    )
    card(
        sl,
        6.8,
        5,
        5.75,
        1.65,
        "Maliyet",
        "NLSS için 1130 s RAM eğitimi bildirildi; belirgin kazanç yok.",
        accent="amber",
    )
    # 08
    sl = slide(
        "07 / Bu ortamda çalıştırılan",
        "İlk Python sonuçları: ortak dinamik ağ.",
        "NN_joint, DDV tahmin hatasını yardımcı eğitim kaybı olarak kullanır.",
    )
    table(
        sl,
        0.75,
        2.7,
        11.8,
        ["Veri seti / RAM Cascade", "Train", "Validation", "Test"],
        [
            ["Mevcut sentetik MAT"] + fits(records["20261112_053325_290124"]),
            ["Darbe–bekleme sentetik"] + fits(records["20261112_053356_805234"]),
        ],
        [5.5, 2.1, 2.1, 2.1],
    )
    card(
        sl,
        0.75,
        4.85,
        5.75,
        1.75,
        "Ne işe yaradı?",
        "Akımdan DDV’ye ve RAM’e ortak hata yayılımı.",
    )
    card(
        sl,
        6.8,
        4.85,
        5.75,
        1.75,
        "Ne garanti değil?",
        "Daha büyük ağ / daha uzun eğitim her zaman daha iyi olmadı.",
    )
    # 09
    sl = slide(
        "08 / Yeni sentetik profil",
        "Uzun tutmalar, drift ve ters darbeler.",
        "FlightLike: 11 uçuş, 2100–2800 s. Parametreler modelden gizli; Truth yalnız audit içindir.",
    )
    chart(
        sl,
        0.75,
        2.55,
        7.7,
        3.85,
        [("Ölçülen RAM", flight[:, 0], flight[:, 3], "dark")],
        target + " · sentetik RAM / zaman (s)",
        (-110, 110),
    )
    card(
        sl,
        8.75,
        2.55,
        3.85,
        3.85,
        "Bilerek zorlaştırıldı",
        "• Uçuşa göre null/band\n• Yavaş sıfır kayması\n• ± darbeler, burst’ler\n• Kısa ve uzun tutma\n• Sonlu hız / strok",
    )
    # 10
    sl = slide(
        "09 / Üç yeni deneme",
        "Validation Fit: %68,13 → %93,47.",
        "Aynı TRAIN/VAL ve donuk DDV/açı blokları; yalnız RAM adayı değişti.",
    )
    trialrows = [("Önceki null izleyici", 68.12975468084734, "muted")] + [
        (t["model"], t["validation_fit"], "teal" if t["model"] == selected else "blue")
        for t in result["trials"]
    ]
    for i, (name, v, c) in enumerate(trialrows):
        yy = 2.65 + i * 0.78
        text(sl, 0.8, yy, 3.4, 0.36, name, 17, "ink")
        box(sl, 4.25, yy + 0.02, 6.8, 0.3, "card", False)
        box(sl, 4.25, yy + 0.02, 6.8 * max(0, v) / 100, 0.3, c, False)
        text(sl, 11.3, yy - 0.05, 1.2, 0.5, f"%{v:.2f}", 20, c, True)
    text(
        sl,
        0.85,
        6.2,
        11.5,
        0.45,
        "Validation %95’i geçmedi. Üç aday sınırında duruldu; test ile ayar seçilmedi.",
        18,
        "amber",
    )
    # 11
    sl = slide(
        "10 / Seçilen mekanizma",
        "Hareketi null kayması sanma.",
        "Rate_envelope: öğrenilmiş hız haritası + sakinlik/tepe takibi + konum hafızası.",
    )
    labels = [
        ("DDVhat", "Akımdan öğrenilmiş\nara sinyal"),
        ("Null izleyici", "Sakin aralıklar +\nçift yönlü tepeler"),
        ("Hız haritası", "TRAIN’den öğrenilen\nPWL, sıfır hız bandı"),
        ("RAM durumu", "Sızıntısız integral\nTRAIN strok sınırları"),
    ]
    for i, (a, b) in enumerate(labels):
        card(sl, 0.7 + i * 3.18, 2.65, 2.97, 2.4, a, b)
    text(
        sl,
        0.85,
        5.65,
        11.7,
        0.65,
        "Simülatörün gerçek null veya fiziksel gain değerleri modele verilmez. Yeni hız ailesi NN değildir.",
        20,
        "teal",
    )
    # 12
    sl = slide(
        "11 / Kayıtlı validation uçuşu",
        "DDV → RAM → açı: ölçüm ve tahmin.",
        "İlk validation uçuşu gösteriliyor; örnek seçiminde test kullanılmadı.",
    )
    for j, (label, column, limits) in enumerate(
        [("DDV", 2, (-0.6, 0.6)), ("RAM", 3, (-110, 110)), ("Açı", 4, (-35, 40))]
    ):
        chart(
            sl,
            0.8,
            2.45 + j * 1.44,
            11.7,
            1.35,
            [
                ("Ölçülen", flight[:, 0], flight[:, column], "dark"),
                ("Rate_envelope", flight[:, 0], new[j], "teal"),
            ],
            label + " · " + target,
            limits,
        )
    # 13
    sl = slide(
        "12 / Sonuç ve sınır",
        "Cascade iyileşti; gürültü hassasiyeti kaldı.",
        "Sentetik sonuçlar — gerçek uçuş kabulü yerine geçmez.",
    )
    table(
        sl,
        0.75,
        2.5,
        11.8,
        ["Seçili model / aşama", "Train", "Validation", "Test"],
        [
            ["RAM Cascade"] + fits(result),
            ["RAM Component"] + fits(result, "RAM_Component"),
            ["Açı Cascade"] + fits(result, "Angle_Cascade"),
        ],
        [5.5, 2.1, 2.1, 2.1],
    )
    fresh = next(
        r["fit_test"]
        for r in result["fresh_test_summary"]
        if r["stage"] == "RAM_Cascade"
    )
    metric(
        sl,
        0.75,
        5.05,
        4.05,
        "İki yeni sentetik uçuş",
        f"%{fresh:.2f}",
        "Seed 9621; seçimden sonra üretildi",
    )
    text(
        sl,
        5.2,
        5.1,
        7.1,
        1.25,
        "Ham DDV gürültüsü observer’ı bozabiliyor.\nEski test daha önce görülmüştü; yeni audit ayrı raporlandı.\nGerçek veriyle tekrar doğrulama gerekli.",
        18,
        "amber",
    )
    # 14
    sl = slide(
        "13 / Çevrimdışı kullanım",
        "Python 3.9. İnternetsiz eğitim bilgisayarı.",
        "start.bat artık paket indirmez; eksikleri bildirir ve durur.",
    )
    card(
        sl,
        0.75,
        2.65,
        3.8,
        2.5,
        "1 · Paket kaynağı",
        "Yerel wheelhouse\nWindows x64 + CPython 3.9\nTüm alt bağımlılıklar",
    )
    card(
        sl,
        4.8,
        2.65,
        3.8,
        2.5,
        "2 · Tek seferlik kurulum",
        "setup_offline.bat\n--no-index\n--find-links yerel klasör",
    )
    card(
        sl,
        8.85,
        2.65,
        3.75,
        2.5,
        "3 · Uygulamayı aç",
        "start.bat\n127.0.0.1:8765\nCSV, grafik, XLSX, modeller",
    )
    text(
        sl,
        0.85,
        5.8,
        11.7,
        0.7,
        "NumPy · SciPy · pandas · openpyxl · PyTorch CPU · FastAPI · Pydantic · Uvicorn",
        18,
        "teal",
    )
    # 15
    sl = slide(
        "14 / Yol haritası",
        "Sonraki adım: sağlamlığı gerçek veriyle sınamak.",
        "Bugünkü ilerleme: sınırlı arama, tekrarlanabilir sonuç ve çevrimdışı çalışabilirlik.",
    )
    for i, (a, b) in enumerate(
        [
            (
                "Önce",
                "Ham DDV gürültüsüne hassasiyeti azalt; hold / hareket hatasını ayrı izle.",
            ),
            (
                "Sonra",
                "Aynı gerçek split ile eğitim ve validation; sentetik eklemeyi yalnız TRAIN’de A/B karşılaştır.",
            ),
            (
                "Dağıtıma giderken",
                "Sabit boyutlu çıkarımı C’ye taşı; 200 Hz arayüz ve S-Function eşdeğerliğini test et.",
            ),
        ]
    ):
        card(sl, 0.75 + i * 4.2, 2.7, 3.95, 3.05, a, b)
    text(
        sl,
        0.85,
        6.15,
        11.6,
        0.45,
        "C/MEX henüz üretilmedi. %95 validation eşiği bu turda elde edilmedi.",
        18,
        "amber",
    )
    # 16
    sl = slide(
        "Ek / Düzenleme ve kaynaklar",
        "Bu sunum kendi templatene taşınabilir.",
        "Metinler, şemalar, tablolar ve grafikler native PowerPoint nesneleridir.",
    )
    card(
        sl,
        0.75,
        2.65,
        5.75,
        3.25,
        "Düzenleme",
        "PowerPoint’te metinleri/şekilleri doğrudan değiştir.\nGrafik → Verileri Düzenle ile gömülü tabloya eriş.\nKendi temana slaytları yeniden kullanarak taşı.",
    )
    card(
        sl,
        6.8,
        2.65,
        5.75,
        3.25,
        "Kaynaklar",
        "docs/changelog.md\ndocs/experiments.md\npython/reference_results/\npython/build_presentation.py",
    )
    text(
        sl,
        0.85,
        6.25,
        11.6,
        0.35,
        "Gerçek uçuş sayıları kullanıcı bildirimidir; sentetik sayılar kayıtlı Python koşularından alınmıştır.",
        15,
        "muted",
    )
    for i, sl in enumerate(prs.slides, 1):
        sl.notes_slide.notes_text_frame.text = f"Slayt {i}. Kaynak: dev_CodexPy docs/changelog.md ve docs/experiments.md. Son deney: {NEW}. Gerçek uçuş sonuçları kullanıcı tarafından bildirilmiştir; bu ortamda gerçek uçuş verisi yoktur. Grafik önizlemeleri extrema koruyarak seyreltilmiştir. Düzenlenebilir kaynak: python/build_presentation.py."
    OUT.parent.mkdir(exist_ok=True)
    prs.save(str(OUT))
    # Structural validation: no accidental off-slide elements; chart workbooks retained.
    for si, sl in enumerate(prs.slides, 1):
        for sh in sl.shapes:
            assert sh.left >= 0 and sh.top >= 0, (si, sh.name)
            assert sh.left + sh.width <= prs.slide_width + Inches(0.02), (
                si,
                sh.name,
                "width",
            )
            assert sh.top + sh.height <= prs.slide_height + Inches(0.02), (
                si,
                sh.name,
                "height",
            )
    print(str(OUT))
    print(
        "Slides:",
        len(prs.slides),
        "native charts:",
        sum(1 for s in prs.slides for sh in s.shapes if sh.has_chart),
    )


if __name__ == "__main__":
    build()
