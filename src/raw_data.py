from singular_data import SingularData
from datetime import datetime

class RawData:
    TIMESTAMP_PATH: str = "timestamps.txt"
    DATAFORMAT_PATH: str = "dataformat.txt"
    DATA_PATH: str = "data"
    
    path: str
    data: list[SingularData]
    
    def __init__(self, path: str) -> None:
        self.path = path
        self.data = []
        self.load_data()
        
    def load_data(self) -> None:
        with open(f"{self.path}/{self.TIMESTAMP_PATH}", 'r') as file:
            timestamps = file.read().splitlines()
        for index, timestamp in enumerate(timestamps):
            timestamp_date = datetime.fromisoformat(timestamp)
            singular_data = SingularData(f"{self.path}/{self.DATA_PATH}", timestamp_date, index)
            self.data.append(singular_data)