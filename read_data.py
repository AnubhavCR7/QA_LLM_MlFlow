import pandas as pd
import logging

# Create and configure logger
logging.basicConfig(filename = "output_logs.log", format = '%(asctime)s %(message)s', filemode = 'a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def read_data(data_path):
    """
    Read Question Answering dataset. Only .csv and .xlsx files are allowed.
    Assumes that the first row of the file is header.
    """

    logger.info(f"  Input Dataset Path : {data_path}")
    print(f"\nInput Dataset Path : {data_path}")
    
    try:
        if data_path.split(".")[-1] == "csv":
            qa_data = pd.read_csv(data_path, header = 0, sep = ',')
        elif data_path.split(".")[-1] == "xlsx":
            qa_data = pd.read_excel(data_path, header = 0)
        else:
            ext = "." + data_path.split(".")[-1]
            text = f"\nOnly .csv and .xlsx files are allowed. Input File has {ext} as extension"
            logger.error("  " + text.strip("\n"))
            raise ValueError(text)
        
        if ("inputs" != qa_data.columns[0]) or ("ground_truth" != qa_data.columns[1]):
            text = "\nInput Dataset should have 'input' as first column and 'ground_truth' as second column. Please make sure the column names align."
            logger.error("  " + text.strip("\n"))
            raise ValueError(text)
        
        msg = f"\nInput Dataset has {qa_data.shape[0]} records and {qa_data.shape[1]} columns."
        logger.info("  " + msg.strip("\n"))
        print(msg)
        return qa_data
    except Exception as e:
        logger.info("  Error in Reading Input Dataset. See below.")
        logger.info(str(e))
        print("\nError in Reading Input Dataset. See below.\n")
        raise
