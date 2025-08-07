import pandas as pd

class ResultAnalyzer:
    @staticmethod
    def save_results(data: list[dict], output_path: str):
        df = pd.DataFrame(data)
        pd.set_option("display.float_format", "{:.3f}".format)
        print(df.head())
        df.to_csv(output_path, index=False)
