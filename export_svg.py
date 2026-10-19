import xml.etree.ElementTree as ET
from pathlib import Path


def export_svg(model, filename="tree.svg", x_spacing=80, y_spacing=20, node_radius=3):
    #on parcourt les layers et les noeuds deja construits par le modele
    columns = []
    trunk = model.root
    while trunk is not None:
        nodes = []
        node = model._get_bottom_node(trunk)
        while node is not None:
            nodes.append(node)
            node = node.up_node
        columns.append((trunk, nodes))
        trunk = trunk.next_trunk

    k_min = min(node.up_count for _, nodes in columns for node in nodes)
    k_max = max(node.up_count for _, nodes in columns for node in nodes)
    width = 160 + model.steps * x_spacing
    height = 240 + (k_max - k_min) * y_spacing
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg",
                     width=str(width), height=str(height), viewBox=f"0 0 {width} {height}")
    ET.SubElement(svg, "rect", width="100%", height="100%", fill="white")
    ET.SubElement(svg, "style").text = (
        "text { font-family: Arial, sans-serif; fill: #334155; font-size: 12px; }"
        ".node:hover { stroke: #0f172a; stroke-width: 2; }"
    )

    def text(x, y, label, size=12, anchor="start"):
        ET.SubElement(svg, "text", x=str(x), y=str(y),
                      attrib={"text-anchor": anchor, "style": f"font-size: {size}px"}).text = label

    text(30, 32, "Arbre trinomial", 22)
    text(30, 56, "Horizontal : layer / date. Vertical : position dans la grille (up_count), pas le prix.")
    for x, color, label in [(30, "#2563eb", "3 branches"),
                             (200, "#ea580c", "Élagué : centre seul"),
                             (430, "#64748b", "Maturité")]:
        ET.SubElement(svg, "circle", cx=str(x + 4), cy="78", r="4", fill=color)
        text(x + 16, 82, label)

    coords = {}
    for trunk, nodes in columns:
        x = 80 + trunk.step * x_spacing
        ET.SubElement(svg, "line", x1=str(x), x2=str(x), y1="145", y2=str(height - 80),
                      stroke="#e2e8f0", attrib={"stroke-dasharray": "3 5"})
        text(x, 112, f"Layer {trunk.step}", anchor="middle")
        date = model.start_date + model.index_dt * trunk.step
        text(x, 130, date.strftime("%d/%m"), anchor="middle")
        text(x, height - 52, f"{len(nodes)} nœuds", anchor="middle")
        dividends = model.dividends_by_step.get(trunk.step, [])
        if dividends:
            text(x, height - 32, f"Div. {sum(d.amount for d in dividends):g}", anchor="middle")
        for node in nodes:
            coords[node] = (x, 160 + (k_max - node.up_count) * y_spacing)

    #on dessine uniquement les liens existants, avant les cercles
    edges = ET.SubElement(svg, "g", id="branches")
    for _, nodes in columns:
        for node in nodes:
            x, y = coords[node]
            pruned = node.next_mid is not None and node.next_up is None and node.next_down is None
            for child, prob in [(node.next_up, node.p_up), (node.next_mid, node.p_mid),
                                (node.next_down, node.p_down)]:
                if child is None:
                    continue
                next_x, next_y = coords[child]
                line = ET.SubElement(edges, "line", x1=str(x), y1=str(y), x2=str(next_x), y2=str(next_y),
                                     stroke="#ea580c" if pruned else "#cbd5e1",
                                     attrib={"stroke-width": "1.8" if pruned else "0.8"})
                ET.SubElement(line, "title").text = f"Probabilité de transition : {prob:.6g}"

    circles = ET.SubElement(svg, "g", id="nodes")
    for trunk, nodes in columns:
        for node in nodes:
            x, y = coords[node]
            pruned = node.next_mid is not None and node.next_up is None and node.next_down is None
            color = "#ea580c" if pruned else "#2563eb"
            if trunk.next_trunk is None:
                color = "#64748b"
            circle = ET.SubElement(circles, "circle", cx=str(x), cy=str(y), r=str(node_radius),
                                   fill=color, attrib={"class": "node"})
            label = f"Layer {node.step} | k = {node.up_count}\nSpot = {node.underlying_price:.6f}"
            if node.opt_price is not None:
                label += f"\nOption = {node.opt_price:.6f}"
            if hasattr(node, "prob"):
                label += f"\nProbabilité depuis la racine = {node.prob:.6g}"
            if pruned:
                label += "\nPruning : (p_up, p_mid, p_down) = (0, 1, 0)"
            ET.SubElement(circle, "title").text = label

    text(30, height - 10, "Survoler un nœud ou une branche pour afficher ses valeurs.")
    ET.ElementTree(svg).write(filename, encoding="utf-8", xml_declaration=True)
    return Path(filename).resolve()

