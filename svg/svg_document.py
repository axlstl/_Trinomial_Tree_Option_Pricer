from pathlib import Path
import xml.etree.ElementTree as ET


class SvgDocument:
    def __init__(self, width: float, height: float, stylesheet: Path) -> None:
        self.root = ET.Element("svg", 
                               xmlns="http://www.w3.org/2000/svg", 
                               width=str(width), 
                               height=str(height),
                               viewBox=f"0 0 {width} {height}")
        ET.SubElement(self.root, "rect", width="100%", height="100%", fill="white")
        ET.SubElement(self.root, "style").text = stylesheet.read_text(encoding="utf-8")

    def text(self, position: tuple[float, float], label: str, style: str = "", anchor: str = "start") -> None:
        x, y = position
        ET.SubElement(self.root, "text", x=str(x), y=str(y),attrib={"class": style, "text-anchor": anchor}).text = label

    def group(self, name: str) -> ET.Element:
        return ET.SubElement(self.root, "g", id=name)

    def line(self, parent: ET.Element, start: tuple[float, float], end: tuple[float, float],
             style: str, title: str = "") -> None:
        """This function is in charge of drawing the line between two nodes 
        Args:
            parent (ET.Element): 
            start (tuple[float, float]): ((x,y) coordinates of the parent node
            end (tuple[float, float]): (x,y) coordinates of the child node 
            style (str): either branch or pruned-branch
            title (str, optional): tooltip (displayed when hovering the branch)
        """
        x1, y1 = start
        x2, y2 = end
        element = ET.SubElement(parent, "line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2),attrib={"class": style})
        self._add_title(element, title)

    def circle(self, parent: ET.Element, center: tuple[float, float], radius: float,style: str, title: str = "") -> None:
        x, y = center
        element = ET.SubElement(parent, "circle", cx=str(x), cy=str(y), r=str(radius), attrib={"class": style})
        self._add_title(element, title)

    def _add_title(self, element: ET.Element, title: str) -> None:
        """Adds a title (infobulle) to an SVG element displayed when hovering the mouse on nodes"""
        if title:
            ET.SubElement(element, "title").text = title

    def write(self, filename: str | Path) -> Path:
        ET.ElementTree(self.root).write(filename, encoding="utf-8", xml_declaration=True)
        return Path(filename).resolve()
