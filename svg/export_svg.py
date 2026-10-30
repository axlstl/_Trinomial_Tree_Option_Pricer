from pathlib import Path
from platform import node
from xml.etree.ElementTree import Element

from model.lattice import LatticeModel

from node import Node, TrunkNode
from svg.svg_document import SvgDocument


SIDE_MARGIN = 80
TOP_MARGIN = 20
BOTTOM_MARGIN = 60
STYLESHEET = Path(__file__).with_name("tree_style.css")

X_SPACING = 50
Y_SPACING = 10
NODE_RADIUS = 5

def export_svg(model: LatticeModel, filename: str | Path = "tree.svg") -> Path:

    if model.root_node.opt_price is None:
        raise ValueError("Please price the option first before trying to export the SVG.")

    if isinstance(filename, str):
        filename = Path(filename)

    drawing = TreeSvg(model)
    return drawing.write(filename)


class TreeSvg:
    """This class is in charge of drawing the tree : where to place nodes, 
        how to build branches, how and where is the text displayed on the document...
    """
    def __init__(self, model: LatticeModel) -> None:
        self.model = model
        self.x_spacing = X_SPACING
        self.y_spacing = Y_SPACING
        self.node_radius = NODE_RADIUS

    def write(self, filename: Path) -> Path:
        """Main function for exporting the svg tree. It will successively :
            1. 'collect' the columns of the tree, meaning it retrieves all the node of each time step
            2. place the nodes in the document
            3. draw the columns, branches and nodes
            4. Save the svg file

        Args:
            filename (Path): Path of the svg file to be created

        Returns:
            Path: Path of the svg tree
        """
        self.columns = self._retrieve_nodes_from_tree()
        self._place_nodes()
        self.svg = SvgDocument(self.width, self.height, STYLESHEET)

        #Once all relative positions are calculated, we draw the tree in the document
        self._draw_columns()
        self._draw_branches()
        self._draw_nodes()
        return self.svg.write(filename)
    
    def _retrieve_nodes_from_tree(self) -> list[tuple[TrunkNode, list[Node]]]:
        """Retrieve all nodes, going from model.root to model.last_trunk, storing them
            in a list of tuples, each tuple containing the trunk node and a list of its nodes.
        Returns:
            list[tuple[TrunkNode, list[Node]]]: A list with the following form : 
                                                [(trunk_date_0, [node_0]),
                                                (trunk_date_1, [node_1, node_2, node_3]),...]
        """
        columns = []
        trunk = self.model.root
        while trunk is not None:
            nodes = []
            #For each layer, retrieving all nodes from bottop to top
            node = self.model._get_bottom_node(trunk)
            while node is not None:
                nodes.append(node)
                node = node.up_node
            columns.append((trunk, nodes))

            trunk = trunk.next_trunk
        return columns

    def _place_nodes(self) -> None:
        """ This function is in charge of calculating nodes x and y coordinates in the svg document
            To do so, it uses the previous columns list built in _retrieve_nodes_from_tree()
            For each node :
                x is calculated from the time step and the x_spacing (representing the horizontal distance between two nodes)
                y is calculated from its relative value to the initial price (a 1% change from S0 is equal to Y_SPACING pixels of vertical distance)
            We also take into account the top and bottom margins for each calculation
        """

        self.positions = {}
  
        for _, nodes in self.columns:
            for node in nodes:

                x = SIDE_MARGIN + node.step * self.x_spacing
                change = 100 * (node.underlying_price / self.model.S0 - 1)
                self.positions[node] = (x, -change * self.y_spacing)

        top = min(y for x, y in self.positions.values())
        bottom = max(y for x, y in self.positions.values())

        self.width = 2 * SIDE_MARGIN + self.model.steps * self.x_spacing
        self.height = TOP_MARGIN + bottom - top + BOTTOM_MARGIN
        # Move the whole drawing inside the page without changing the scale.
        for node, (x, y) in self.positions.items():
            self.positions[node] = (x, TOP_MARGIN + y - top)




    def _draw_columns(self) -> None:
        for trunk, nodes in self.columns:
            x = SIDE_MARGIN + trunk.step * self.x_spacing
            # Draw a vertical line at each time step.
            self.svg.line(self.svg.root, (x, TOP_MARGIN), (x, self.height - BOTTOM_MARGIN), "grid")
            self._draw_column_labels(trunk, nodes, x)

    def _draw_column_labels(self, trunk: TrunkNode, nodes: list[Node], x: float) -> None:
        """This function is in charge of writing the small text under each column (curr number of nodes + div amount)
        Args:
            trunk (TrunkNode): The trunk node of the column
            nodes (list[Node]): The list of nodes in the corresponding trunk column
            x (float): The x coordinate of the corresponding node column in the svg document 
        """
        # 1. We display the number of nodes every 5 layers and on the last one
        if trunk.step % 5 == 0 or trunk.next_trunk is None:
            self.svg.text((x, self.height - 35), f"{len(nodes)} nœuds", anchor="middle")
        # 2. If theres a dividend, we also display the amount
        dividends = self.model.dividends_by_step.get(trunk.step, [])
        if dividends:
            amount = sum(dividend.amount for dividend in dividends)
            self.svg.text((x, self.height - 15), f"Div. {amount:g}", anchor="middle")

    def _draw_branches(self) -> None:
        """This function is in charge of drawing the lines between the nodes
        """
        group = self.svg.group("branches")
        for node in self.positions:
            children = [node.next_up, node.next_mid, node.next_down]
            style = "pruned-branch" if self._is_pruned(node) else "branch"
            # once we know if the branch is pruned or not, we draw the line between the node and its children
            for child in children:
                if child is not None:
                    self.svg.line(group, self.positions[node], self.positions[child], style)

    def _draw_nodes(self) -> None:
        """This function is in charge of drawing the nodes circles + the tooltip text
        """
        group = self.svg.group("nodes")
        for node, position in self.positions.items():
            self._draw_node(group, node, position)

    def _draw_node(self, group: Element, node: Node, position: tuple[float, float]) -> None:
        style = "regular"
        if self._is_pruned(node):
            style = "pruned"
        if node.trunk_node.next_trunk is None:
            style = "maturity"
        # Add a larger red circle if early exercise is optimal at this node.
        if self.model.option.exercise_type == "american" and node.exercise_now:
            self.svg.circle(group, position, self.node_radius + 3, "exercise")
        # Draw the point and add the values displayed when hovering over it.
        label = self._node_label(node)
        self.svg.circle(group, position, self.node_radius, "node " + style, label)

    def _node_label(self, node: Node) -> str:
        label = (
                f"Noeud {node.step} / k = {node.up_count}\n"
                f"Spot = {node.underlying_price:.2f}\n"
                f"Prix de l'option = {node.opt_price:.2f}\n"
                f"Probabilité cumulée = {node.cum_proba:.4g}"
            )
        # Add the pruning and exercise information when it applies.
        if self.model.option.exercise_type == "american" and node.exercise_now:
            label += "\nExercer l'option de manière anticipée est optimal à cet endroit"
        return label

    def _is_pruned(self, node: Node) -> bool:
        return (node.next_mid is not None) and (node.next_up is None and node.next_down is None)
