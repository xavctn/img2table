# coding: utf-8
from typing import List

from img2table.tables.objects.cell import Cell
from img2table.tables.processing.borderless_tables.model import TableLine, LineGroup, ImageSegment


def identify_lines(elements: List[Cell], ref_size: int) -> List[TableLine]:
    """
    Identify lines from Cell elements
    :param elements: list of cells
    :param ref_size: reference distance between two line centers
    :return: list of table lines
    """
    if len(elements) == 0:
        return []

    elements = sorted(elements, key=lambda c: c.y1 + c.y2)

    # Group elements in lines
    seq = iter(elements)
    tb_lines = [TableLine(cells=[next(seq)])]
    for cell in seq:
        if (cell.y1 + cell.y2) / 2 - tb_lines[-1].v_center > ref_size:
            tb_lines.append(TableLine(cells=[]))
        tb_lines[-1].add(cell)

    # Remove overlapping lines
    dedup_lines = list()
    for line in tb_lines:
        # Get number of overlapping lines
        overlap_lines = [l for l in tb_lines if line.overlaps(l) and not line == l]

        if len(overlap_lines) <= 1:
            dedup_lines.append(line)

    # Merge lines that corresponds
    merged_lines = [[l for l in dedup_lines if line.overlaps(l)] for line in dedup_lines]
    merged_lines = [line.pop() if len(line) == 1 else line[0].merge(line[1]) for line in merged_lines]

    return list(set(merged_lines))


def create_h_pos_groups(lines: List[TableLine]) -> List[List[TableLine]]:
    """
    Create group of lines based on their horizontal position
    :param lines: list of TableLine objects
    :return: groups of lines
    """
    # Loop over all lines to create relationships between horizontally aligned lines
    clusters = list()
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            h_coherent = min(lines[i].x2, lines[j].x2) - max(lines[i].x1, lines[j].x1) > 0
            # If lines are horizontally coherent, find matching clusters
            if h_coherent:
                matching_clusters = [idx for idx, cl in enumerate(clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(clusters) if idx in matching_clusters])
                    clusters = remaining_clusters + [new_cluster]
                else:
                    clusters.append({i, j})

    # Return groups of lines
    line_groups = [[lines[idx] for idx in cl] for cl in clusters]

    return line_groups


def vertically_coherent_groups(lines: List[TableLine], max_gap: float) -> List[LineGroup]:
    """
    Cluster lines into vertically coherent groups
    :param lines: list of lines as TableLine objects
    :param max_gap: maximum gap allowed between consecutive lines in order to be clustered in same group
    :return: list of line groups as LineGroup objects
    """
    # Sort lines by vertical position
    lines = sorted(lines, key=lambda line: line.y1 + line.y2)

    seq = iter(lines)
    v_line_groups = [LineGroup(lines=[next(seq)])]
    for line in seq:
        # If gap between lines is too large, split into new group
        if line.y1 - v_line_groups[-1].y2 > max_gap:
            v_line_groups.append(LineGroup(lines=[]))
        v_line_groups[-1].add(line)

    # Check coherency in each group
    line_groups = list()
    for gp in [gp for gp in v_line_groups if gp.size > 1]:
        seq = iter(gp.lines)
        gps = [LineGroup(lines=[next(seq)])]
        for line in seq:
            line_gap = line.y1 - gps[-1].lines[-1].y2
            line_sep = line.v_center - gps[-1].lines[-1].v_center
            if (line_gap > 1.2 * gp.median_line_gap) and (line_sep > 1.2 * gp.median_line_sep):
                gps.append(LineGroup(lines=[]))
            gps[-1].add(line)

        # Add groups to line groups
        line_groups += [gp for gp in gps if max([len(line.cells) for line in gp.lines]) > 1]

    return line_groups


def is_text_block(line_group: LineGroup, char_length: float) -> bool:
    """
    Check if the line group corresponds to a text block
    :param line_group: LineGroup object
    :param char_length: average character length
    :return: boolean indicating if the line group is a text block
    """
    # Get list of lines and if they are complete text
    nb_lines_text = 0

    for line in line_group.lines:
        text_line = True
        # Check difference in pixels between each element of the line
        cells = sorted(line.cells, key=lambda c: (c.x1, c.x2))

        for prev, nextt in zip(cells, cells[1:]):
            x_diff = nextt.x1 - prev.x2
            if x_diff >= 2 * char_length:
                text_line = False
                break

        nb_lines_text += int(text_line)

    return nb_lines_text / len(line_group.lines) >= 0.5


def identify_line_groups(segment: ImageSegment, char_length: float, median_line_sep: float) -> ImageSegment:
    """
    From elements of the segment, identify lines that are coherent together
    :param segment: ImageSegment object
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: segment with its groups of lines
    """
    # Identify lines in segment
    lines = identify_lines(elements=segment.elements,
                           ref_size=int(median_line_sep // 4))

    # Create line groups
    line_groups = [line_group for h_group in create_h_pos_groups(lines=lines)
                   for line_group in vertically_coherent_groups(lines=h_group, max_gap=median_line_sep)
                   if line_group.size > 1 and not is_text_block(line_group=line_group, char_length=char_length)]

    return ImageSegment(x1=segment.x1,
                        y1=segment.y1,
                        x2=segment.x2,
                        y2=segment.y2,
                        elements=segment.elements,
                        line_groups=line_groups)
