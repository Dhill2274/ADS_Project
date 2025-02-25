from urllib.request import urlretrieve
import os
import zipfile
from pathlib import Path
import logging
import pandas

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)


def main():
    # the following dataset contains countries that took the survey across 2002 to 2023
    # i just manually picked the countries on the dataset builder page
    # i've also just selected the 'politics' variables (we can change this if needed)
    # url: https://ess.sikt.no/en/data-builder/?tab=round_country&rounds=0.2_9-11_13_15_23-38+1.2_9-11_13_15_23-27_32-5_38+2.2_9-27_32-35_38+3.2_9-11_13_15_23-27_32-35_38+4.2_9-11_13_15_23-27_32-35_38+5.2_9-13_15_23-27_32-35_38+6.2_9-15_23-38+7.2_9-13_15_23-27_32-38+8.2_9-13_15_23-27_32-38+9.2_9_10_13_15_23_25_27_32-38+10.11_26_33_34+11.2_9-11_13_15_23-27_32-38&seriesVersion=883&variables=1
    dataset = Path(
        "datasets/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_5-ESS6e02_6-ESS7e02_3-ESS8e02_3-ESS9e03_2-ESS10-ESS10SC-ESS11-subset.csv"
    )

    if not dataset.is_file():
        logger.info("Dataset is not downloaded. Downloading now.")
        dataset_url = "https://stessdissprodwe.blob.core.windows.net/data/download/4/generate_datafile/fc18b3a481661f4b4f99077b940b2111.zip?st=2025-02-25T15%3A27%3A34Z&se=2025-02-25T16%3A29%3A34Z&sp=r&sv=2023-11-03&sr=b&skoid=1b26ad26-8999-4f74-9562-ad1c57749956&sktid=93a182aa-d7bd-4a74-9fb1-84df14cae517&skt=2025-02-25T15%3A27%3A34Z&ske=2025-02-25T16%3A29%3A34Z&sks=b&skv=2023-11-03&sig=/wy/ALXKry6/TRc2lDEiZUYciPSglGeRgilvAnWHB0g%3D"
        dst = "datasets/ess_politics.zip"
        _ = urlretrieve(dataset_url, dst)

        # unzip and clean up
        with zipfile.ZipFile(dst, "r") as zip_ref:
            file_names = []
            files = zip_ref.infolist()
            zip_len = len(files)

            for i in range(zip_len):
                file_names.append(files[i].filename)

            zip_ref.extractall("datasets/")
            os.remove(dst)

            html_files = [file for file in file_names if "html" in file]
            for i in range(len(html_files)):
                os.remove(f"datasets/{html_files[0]}")

    dataset_df = pandas.read_csv(dataset)
    logger.info("ESS Dataset loaded")


if __name__ == "__main__":
    main()
