from src.reporting import layout


def test_kpi_table_renders_named_columns():
    cols = [("Gross", {"ann_return": 0.12, "sharpe": 1.2}),
            ("Net @1×", {"ann_return": 0.016, "sharpe": 0.84})]
    html = layout.kpi_table(cols)
    assert "Gross" in html and "Net @1×" in html
    assert "Annualized return" in html and "Sharpe" in html
    assert "12.0%" in html and "0.84" in html
    assert "—" in html            # missing keys (e.g. sortino) degrade gracefully


def test_data_table_html_cells_passthrough():
    html = layout.data_table(["Year", "net"], [["2021", "<span class='pos'>+31%</span>"]])
    assert "<span class='pos'>+31%</span>" in html and "2021" in html


def test_page_is_self_contained_doc():
    html = layout.page("Title", "subtitle", "<div>cards</div>", "<p>body</p>", "footer")
    assert html.startswith("<!doctype html>")
    assert "Title" in html and "body" in html and "footer" in html
    assert "src=\"http" not in html and "<link" not in html
