import pandas as pd

class DataLoader:
    @staticmethod
    def load_sms_data(path: str) -> list[str]:
        df = pd.read_csv(
            path,
            encoding="latin1",
            usecols=[0,1],
            names=["label","text"],
            header=0
        )
        return df["text"].str.lower().tolist()
