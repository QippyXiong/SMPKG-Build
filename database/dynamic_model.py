r"""
提供构建动态结点和关系的能力。

首先，我们需要规定一个存储协议，来存储我们目前数据库的状况。我们需要存储的信息有：
1. 具有的结点类型
2. 结点类型对应的关系类型

然后，我们需要一个数据库driver，来提供对数据库的操作能力。这个driver需要提供的能力有：

基本类型接口：

Node:
    cls: NodeCls
    attribs: dict
    relations: list[RelationCls]

Relation:
    cls: NodeCls
    begin_node: Node
    end_node: Node
    attibs: dict

1. list_node_cls() --> list[NodeCls]
2. create_node_cls(node_cls: NodeCls) --> bool: 创建一个结点类型，返回是否创建成功
3. list_relation_cls() --> list[RelationCls]
2. create_node(node_cls: NodeCls, attribs: dict) --> bool: 创建一个结点，返回是否创建成功
3. create_relation(relation_cls: RelationCls, begin_node: Node, end_node: Node, attribs: dict) --> bool: 创建一个关系，返回是否创建成功

"""