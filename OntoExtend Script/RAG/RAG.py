from RAG2 import RAG
import sys
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run OntoExtend with custom dataset, CQ, and other options."
    )

    # Optional argument for the Competency Question
    parser.add_argument(
        "-cq", #"--CQ",
        default="What are the components of a product?",
        help="Competency Question with or without a story. Example: 'What are the components of a product?'"
    )

    # Optional argument for starting RAG
    parser.add_argument(
        "-start_rag",
        default=True,
        type=lambda x: str(x).lower() in ["true", "1", "yes", "y", "t"],
        help="Set to True to initialize the RAG embedding space; False to skip."
    )

    parser.add_argument(
        "-core",# "--core",
        default=" ",
        help="Create a folder for your ontology files, put it in dataset, just put its name. e.g. OntoDESIDECoreOntology. all files in this folder without adding folder to this."
    )

    parser.add_argument(
        "-class_count",# "--class_count",
        type=int,
        default=15,
        help="number of classes that RAG returns to you, 13-18 is a good number"
    )
    parser.add_argument(
        "-dp_count",# "--dp_count",
        type=int,
        default=5,
        help="number of data properties that RAG returns to you, 2-6 is a good number"
    )
    parser.add_argument(
        "-op_count",# "--op_count",
        type=int,
        default=3,
        help="number of object properties that RAG returns to you, 4-8 is a good number"
    )

    args = parser.parse_args()
    print(args)
    core = args.core
    competency_question = args.cq
    start_rag = args.start_rag
    class_count = args.class_count
    op_count = args.op_count
    dp_count = args.dp_count

    RAG.RAG(Query=competency_question,init_rag_flag=start_rag,
        class_count=class_count, op_count=op_count, dp_count=dp_count,core=core)

