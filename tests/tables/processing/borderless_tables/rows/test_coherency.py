# coding: utf-8

import json

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import DelimiterGroup, TableRow
from img2table.tables.processing.borderless_tables.rows.coherency import identify_content_alignment, Alignment, \
    is_line_coherent, check_extremity_lines, check_coherency_rows


def test_identify_content_alignment():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    with open("test_data/rows.json", "r") as f:
        table_rows = [TableRow(cells=[Cell(**c) for c in row]) for row in json.load(f)]

    result = identify_content_alignment(delimiter_group=delimiter_group,
                                        table_rows=table_rows)

    expected = [Alignment(type='left',
                          value=133.9375,
                          delim_left=Cell(x1=53, y1=45, x2=93, y2=1147),
                          delim_right=Cell(x1=226, y1=45, x2=233, y2=1147)),
                Alignment(type='center',
                          value=322.25,
                          delim_left=Cell(x1=226, y1=45, x2=233, y2=1147),
                          delim_right=Cell(x1=342, y1=45, x2=348, y2=1147)),
                Alignment(type='center',
                          value=409.0,
                          delim_left=Cell(x1=342, y1=45, x2=348, y2=1147),
                          delim_right=Cell(x1=470, y1=45, x2=565, y2=1147)),
                Alignment(type='left',
                          value=565.0625,
                          delim_left=Cell(x1=470, y1=45, x2=565, y2=1147),
                          delim_right=Cell(x1=651, y1=45, x2=724, y2=1147)),
                Alignment(type='center',
                          value=748.125,
                          delim_left=Cell(x1=651, y1=45, x2=724, y2=1147),
                          delim_right=Cell(x1=776, y1=45, x2=841, y2=1147)),
                Alignment(type='center',
                          value=912.5,
                          delim_left=Cell(x1=776, y1=45, x2=841, y2=1147),
                          delim_right=Cell(x1=978, y1=45, x2=999, y2=1121)),
                Alignment(type='center',
                          value=1101.375,
                          delim_left=Cell(x1=978, y1=45, x2=999, y2=1121),
                          delim_right=Cell(x1=1129, y1=45, x2=1133, y2=1121)),
                Alignment(type='center',
                          value=1144.5,
                          delim_left=Cell(x1=1129, y1=45, x2=1133, y2=1121),
                          delim_right=Cell(x1=1233, y1=45, x2=1277, y2=1147))
                ]

    assert result == expected


def test_is_line_coherent():
    alignments = [Alignment(type='left',
                            value=133.9375,
                            delim_left=Cell(x1=53, y1=45, x2=93, y2=1147),
                            delim_right=Cell(x1=226, y1=45, x2=233, y2=1147)),
                  Alignment(type='center',
                            value=322.25,
                            delim_left=Cell(x1=226, y1=45, x2=233, y2=1147),
                            delim_right=Cell(x1=342, y1=45, x2=348, y2=1147)),
                  Alignment(type='center',
                            value=409.0,
                            delim_left=Cell(x1=342, y1=45, x2=348, y2=1147),
                            delim_right=Cell(x1=470, y1=45, x2=565, y2=1147)),
                  Alignment(type='left',
                            value=565.0625,
                            delim_left=Cell(x1=470, y1=45, x2=565, y2=1147),
                            delim_right=Cell(x1=651, y1=45, x2=724, y2=1147)),
                  Alignment(type='center',
                            value=748.125,
                            delim_left=Cell(x1=651, y1=45, x2=724, y2=1147),
                            delim_right=Cell(x1=776, y1=45, x2=841, y2=1147)),
                  Alignment(type='center',
                            value=912.5,
                            delim_left=Cell(x1=776, y1=45, x2=841, y2=1147),
                            delim_right=Cell(x1=978, y1=45, x2=999, y2=1121)),
                  Alignment(type='center',
                            value=1101.375,
                            delim_left=Cell(x1=978, y1=45, x2=999, y2=1121),
                            delim_right=Cell(x1=1129, y1=45, x2=1133, y2=1121)),
                  Alignment(type='center',
                            value=1144.5,
                            delim_left=Cell(x1=1129, y1=45, x2=1133, y2=1121),
                            delim_right=Cell(x1=1233, y1=45, x2=1277, y2=1147))
                  ]

    with open("test_data/rows.json", "r") as f:
        table_rows = sorted([TableRow(cells=[Cell(**c) for c in row]) for row in json.load(f)], key=lambda r: r.y1)

    assert is_line_coherent(row=table_rows[0], column_alignments=alignments)
    assert not is_line_coherent(row=table_rows[-1], column_alignments=alignments)


def test_check_extremity_lines():
    alignments = [Alignment(type='left',
                            value=133.9375,
                            delim_left=Cell(x1=53, y1=45, x2=93, y2=1147),
                            delim_right=Cell(x1=226, y1=45, x2=233, y2=1147)),
                  Alignment(type='center',
                            value=322.25,
                            delim_left=Cell(x1=226, y1=45, x2=233, y2=1147),
                            delim_right=Cell(x1=342, y1=45, x2=348, y2=1147)),
                  Alignment(type='center',
                            value=409.0,
                            delim_left=Cell(x1=342, y1=45, x2=348, y2=1147),
                            delim_right=Cell(x1=470, y1=45, x2=565, y2=1147)),
                  Alignment(type='left',
                            value=565.0625,
                            delim_left=Cell(x1=470, y1=45, x2=565, y2=1147),
                            delim_right=Cell(x1=651, y1=45, x2=724, y2=1147)),
                  Alignment(type='center',
                            value=748.125,
                            delim_left=Cell(x1=651, y1=45, x2=724, y2=1147),
                            delim_right=Cell(x1=776, y1=45, x2=841, y2=1147)),
                  Alignment(type='center',
                            value=912.5,
                            delim_left=Cell(x1=776, y1=45, x2=841, y2=1147),
                            delim_right=Cell(x1=978, y1=45, x2=999, y2=1121)),
                  Alignment(type='center',
                            value=1101.375,
                            delim_left=Cell(x1=978, y1=45, x2=999, y2=1121),
                            delim_right=Cell(x1=1129, y1=45, x2=1133, y2=1121)),
                  Alignment(type='center',
                            value=1144.5,
                            delim_left=Cell(x1=1129, y1=45, x2=1133, y2=1121),
                            delim_right=Cell(x1=1233, y1=45, x2=1277, y2=1147))
                  ]

    with open("test_data/rows.json", "r") as f:
        table_rows = sorted([TableRow(cells=[Cell(**c) for c in row]) for row in json.load(f)], key=lambda r: r.y1)

    result = check_extremity_lines(table_rows=table_rows,
                                   alignments=alignments,
                                   median_row_sep=66)

    assert set(result) == set(table_rows[:-1])


def test_check_coherency_rows():
    with open("test_data/delimiter_group.json", "r") as f:
        data = json.load(f)
    delimiter_group = DelimiterGroup(delimiters=[Cell(**c) for c in data.get('delimiters')],
                                     elements=[Cell(**c) for c in data.get('elements')])

    with open("test_data/rows.json", "r") as f:
        table_rows = sorted([TableRow(cells=[Cell(**c) for c in row]) for row in json.load(f)], key=lambda r: r.y1)

    result = check_coherency_rows(delimiter_group=delimiter_group,
                                  table_rows=table_rows,
                                  median_row_sep=66)

    assert set(result) == set(table_rows[:-1])
