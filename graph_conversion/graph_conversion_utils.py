import networkx as nx
import re
import time

from Node import FlatNode


def parse_file(file_name):
    with open(file_name) as f:
        data = f.read()
        data = data.replace("\n", "")
        data = data.replace("\'", '')
        datas = data.split(";")
    headers = []
    start_data = 0
    for i, line in enumerate(datas):
        if line != 'DATA':
            headers.append(line)
        else:
            start_data = i + 1
            break
    for i in range(0, start_data):
        datas.pop(0)
    return headers, datas


def get_data_section_from_step(step_path):
    datas = []
    with open(step_path) as file:
        data_started = False
        for line in file.readlines():
            if line.startswith("DATA"):
                data_started = True
                continue
            if data_started:
                if line and not line.startswith("ENDSEC"):
                    datas.append(line.strip().replace("'", ""))
                else:
                    break
    return datas


def split_composed_arguments(string):
    parentesis_count = 0
    composed_arguments = []
    start = 0
    end = 0
    for idx, char1 in enumerate(string):
        if char1 == '(':
            parentesis_count += 1
        elif char1 == ')':
            parentesis_count -= 1
            if parentesis_count == 0:
                end = idx + 1
                comp_arg = string[start:end]
                # comp_arg = remove_first_last_space(comp_arg)
                comp_arg = comp_arg.strip()
                composed_arguments.append(comp_arg)
                start = idx + 1
    return composed_arguments


def replace_nodes(all_nodes, fast_dict_search):
    """
    Replace the arguments of the id of neighbour nodes with the neighbour nodes itself. In this way we create archs in the graph.
    Args:
        all_nodes: All FlatNodes with id of their neighbour nodes in the parameters fields.
    """
    for node in all_nodes:
        for index, para in enumerate(node.parameters):
            if isinstance(para, str) and len(para) > 0 and para.startswith(
                    "#"):
                if "-" not in para:
                    node.parameters[index] = fast_dict_search[para]
                else:
                    print("Weird id va --- ")
                # node.parameters[i1] = find_node_by_id(all_nodes, par1)
            # elif isinstance(par1, list):
            #     for i2, par2 in enumerate(par1):
            #         if isinstance(par2, str) and len(par2) > 0 and par2[
            #             0] == '#':
            #             node.parameters[i1][i2] = fast_dict_search[par2]
            #         elif isinstance(par2, list):
            #             for i3, par3 in enumerate(par2):
            #                 if isinstance(par3, str) and len(par3) > 0 and \
            #                         par3[0] == '#':
            #                     node.parameters[i1][i2][i3] = fast_dict_search[
            #                         par3]


def get_nodes_from_datas(datas):
    """
    Generate nodes and archs from a list of lines that compose a .stl file. Each node is characterized by a type and a list of arguments.
    Args:
        datas: list of lines that compose the .stl file.
    Returns:
        all_flat_nodes: list of FlatNode generated from data.
    """
    fast_dict_search = {}
    all_flat_nodes = []
    composed_id = 0
    pattern = r'#\d+'
    type_arguments = None
    node_id = None
    node_type = None
    for line in datas:
        try:
            if line == 'ENDSEC':
                continue
            id_type_arguments = line.split("=")
            if len(id_type_arguments) < 2:
                continue
            node_id = id_type_arguments[0].strip()  # Remove leading and trailing spaces
            type_arguments = id_type_arguments[1].strip()
            # id = remove_first_last_space(id)
            # type_arguments = remove_first_last_space(type_arguments)
            node_type = None
            if type_arguments[0] != '(':  # Normal elements
                type_arguments = type_arguments.split("(", 1)
                node_type = type_arguments[0]
                arguments = type_arguments[1]
                # arguments, _ = split_recursive(arguments, 0)

                arguments = re.findall(pattern, arguments)
                # if number_of_printed_lines < max_lines:
                #     print(f" id: {id} - type_arguments: {type_arguments}")
                #     print(f" type: {type} - arguments: {arguments}")
                #     number_of_printed_lines += 1

            else:  # Composed elements
                multiple_obj = type_arguments[1:][:-1]
                multiple_obj = split_composed_arguments(multiple_obj)
                arguments = []
                for i, m in enumerate(multiple_obj):
                    m_type_arguments = m.split('(', 1)
                    m_type = m_type_arguments[0]
                    m_arguments = m_type_arguments[1]
                    m_arguments = re.findall(pattern, m_arguments)
                    m_id = '##' + str(composed_id)
                    composed_id += 1
                    # node = GenericNode(m_id, m_type, m_arguments)
                    flat_node = FlatNode(m_id, m_type, m_arguments)
                    # all_nodes.append(node)
                    all_flat_nodes.append(flat_node)
                    fast_dict_search[m_id] = flat_node
                    # Composed object properties
                    if i == 0:
                        node_type = 'COMPOSED_' + m_type
                    arguments.append(m_id)

            # node = GenericNode(id, type, arguments)
            flat_node = FlatNode(node_id, node_type, arguments)
            fast_dict_search[node_id] = flat_node
            # all_nodes.append(node)
            all_flat_nodes.append(flat_node)
        except Exception as e:
            print(f"Error while parsing in the second part: {e}")
            print(line)

    # print("   Number of composed id: " + str(composed_id))
    return all_flat_nodes, fast_dict_search


def all_nodes_to_graph(G, all_nodes, name, graph_saves_paths):
    """
    Generate (write in G) a Direct Graph with linked nodes. Each node is characterized by numeric/not numeric parameters.
    Args:
        G: empty graph.
        all_nodes: list of FlatNodes. Each FlatNode has a ID, Type and a list parameters those include numeric
                   attributes, not numeric attributes and neighbour nodes.
        name: name of the graph
        graph_saves_paths: directory where to save the graph.
    Returns: None
    """
    start_time = time.time()
    for flat_node in all_nodes:
        # G.add_node(flat_node)
        dict_protery = FlatNode.get_dict_parameters(flat_node)
        G.add_nodes_from([(flat_node.id, dict_protery)])
    for flat_node in all_nodes:
        numeric_paramenters = []
        for par in flat_node.parameters:
            if isinstance(par, FlatNode):
                G.add_edge(flat_node.id, par.id)
            else:
                numeric_paramenters.append(par)
        flat_node.parameters = numeric_paramenters

    # print("   Graphh of " + name + " model realizing time: %s seconds" % (
    #             time.time() - start_time))
    nx.write_graphml(G, graph_saves_paths)
