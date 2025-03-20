import pandas as pd
import time
import concurrent.futures
import numpy as np
from collections import defaultdict
import functools

# Add a global mapping for dataset display names
DATASET_DISPLAY_NAMES = {
    "chars": "Personal and household characteristics",
    "media": "Media use and trust",
    "politics": "Politics",
    "socio": "Socio-demographics",
    "values": "Human values",
    "wellbeing": "Well-being, exclusion, religion, discrimination, identity"
}

class Dataset:
    def __init__(self, name):
        self.name = name
        self.display_name = DATASET_DISPLAY_NAMES.get(name, name.capitalize())
        print(f"Loading dataset: {name} ({self.display_name})")
        self.df = pd.read_csv(f"cleaned/ess/{name}-cleaned.csv", low_memory=False)
        self.clean_special_values()
        self.questionLabels = self.df.columns.tolist()[3:]
        self.countryLabels = self.df["cntry"].unique().tolist()
        self.rounds = self.df["essround"].unique().tolist()

        # Initialize cache for processed questions
        self._question_cache = {}
        self._mean_cache = {}

        # Dictionary for country data to avoid re-filtering
        self.country_data_dict = {country: self.df[self.df["cntry"] == country] for country in self.countryLabels}

        # Apply question mapping if available
        self._setup_question_mapping(name)
        self.questionLabels = [self.question_mapping.get(q, q) for q in self.questionLabels]
        print(f"Dataset {name} initialized with {len(self.questionLabels)} questions")

    def _setup_question_mapping(self, name):
        """Set up the mapping of question codes to human-readable labels"""
        self.question_mapping = {}
        if name == "politics":
            self.question_mapping = {
                "actrolga": "Able to take active role in political group",
                "badge": "Worn or displayed campaign badge/sticker last 12 months",
                "bctprd": "Boycotted certain products last 12 months",
                "bghtprd": "Bought product for political/ethical/environment reason last 12 months",
                "clsprty": "Feel closer to a particular party than all other parties",
                "contplt": "Contacted politician or government official last 12 months",
                "cptppola": "Confident in own ability to participate in politics",
                "euftf": "European Union: European unification go further or gone too far",
                "freehms": "Gays and lesbians free to live life as they wish",
                "gincdif": "Government should reduce differences in income levels",
                "hmsacld": "Gay and lesbian couples right to adopt children",
                "hmsfmlsh": "Ashamed if close family member gay or lesbian",
                "lrscale": "Placement on left right scale",
                "mmbprty": "Member of political party",
                "pbldmn": "Taken part in lawful public demonstration last 12 months",
                "pbldmna": "Taken part in public demonstration last 12 months",
                "polcmpl": "Politics too complicated to understand",
                "poldcs": "Making mind up about political issues",
                "polintr": "How interested in politics",
                "prtdgcl": "How close to party",
                "prtyban": "Ban political parties that wish overthrow democracy",
                "psppipla": "Political system allows people to have influence on politics",
                "psppsgva": "Political system allows people to have a say in what government does",
                "pstplonl": "Posted or shared anything about politics online last 12 months",
                "ptcpplt": "Politicians care what people think",
                "scnsenv": "Modern science can be relied on to solve environmental problems",
                "sgnptit": "Signed petition last 12 months",
                "stfdem": "How satisfied with the way democracy works in country",
                "stfeco": "How satisfied with present state of economy in country",
                "stfedu": "State of education in country nowadays",
                "stfgov": "How satisfied with the national government",
                "stfhlth": "State of health services in country nowadays",
                "stflife": "How satisfied with life as a whole",
                "trstep": "Trust in the European Parliament",
                "trstlgl": "Trust in the legal system",
                "trstplc": "Trust in the police",
                "trstplt": "Trust in politicians",
                "trstprl": "Trust in country's parliament",
                "trstprt": "Trust in political parties",
                "trstun": "Trust in the United Nations",
                "vote": "Voted last national election",
                "wrkorg": "Worked in another organisation or association last 12 months",
                "wrkprty": "Worked in political party or action group last 12 months",
                "imsmetn": "Allow many/few immigrants of same race/ethnic group as majority",
                "imdfetn": "Allow many/few immigrants of different race/ethnic group from majority",
                "impcntr": "Allow many/few immigrants from poorer countries outside Europe",
                "imbgeco": "Immigration bad or good for country's economy",
                "imueclt": "Country's cultural life undermined or enriched by immigrants",
                "imwbcnt": "Immigrants make country worse or better place to live"
            }
        elif name == "values":
            self.question_mapping = {
                "impdiff": "Important to try new and different things in life",
                "impenv": "Important to care for nature and environment",
                "impfree": "Important to make own decisions and be free",
                "impfun": "Important to seek fun and things that give pleasure",
                "imprich": "Important to be rich, have money and expensive things",
                "impsafe": "Important to live in secure and safe surroundings",
                "imptrad": "Important to follow traditions and customs",
                "ipadvnt": "Important to seek adventures and have an exciting life",
                "ipbhprp": "Important to behave properly",
                "ipcrtiv": "Important to think new ideas and being creative",
                "ipeqopt": "Important that people are treated equally and have equal opportunities",
                "ipfrule": "Important to do what is told and follow rules",
                "ipgdtim": "Important to have a good time",
                "iphlppl": "Important to help people and care for others well-being",
                "iplylfr": "Important to be loyal to friends and devote to people close",
                "ipmodst": "Important to be humble and modest, not draw attention",
                "iprspot": "Important to get respect from others",
                "ipshabt": "Important to show abilities and be admired",
                "ipstrgv": "Important that government is strong and ensures safety",
                "ipsuces": "Important to be successful and that people recognize achievements",
                "ipudrst": "Important to understand different people"
            }
        elif name == "socio":
            self.question_mapping = {
                "anctry1": "First ancestry, European Standard Classification of Cultural and Ethnic Groups",
                "anctry2": "Second ancestry, European Standard Classification of Cultural and Ethnic Groups",
                "atncrse": "Improve knowledge/skills: course/lecture/conference, last 12 months",
                "brwmny": "Borrow money to make ends meet, difficult or easy",
                "chldhhe": "Ever had children living in household",
                "chldhm": "Children living at home or not",
                "cmsrv": "Doing last 7 days: community or military service",
                "cmsrvp": "Partner doing last 7 days: community or military service",
                "crpdwk": "Control paid work last 7 days",
                "crpdwkp": "Partner, control paid work last 7 days",
                "dngdk": "Doing last 7 days: don't know",
                "dngdkp": "Partner doing last 7 days: don't know",
                "dngna": "Doing last 7 days: no answer",
                "dngnap": "Partner doing last 7 days: no answer",
                "dngnapp": "Partner doing last 7 days: not applicable",
                "dngoth": "Doing last 7 days: other",
                "dngothp": "Partner doing last 7 days: other",
                "dngref": "Doing last 7 days: refusal",
                "dngrefp": "Partner doing last 7 days: refusal",
                "domicil": "Domicile, respondent's description",
                "dsbld": "Doing last 7 days: permanently sick or disabled",
                "dsbldp": "Partner doing last 7 days: permanently sick or disabled",
                "dvrcdev": "Ever been divorced",
                "dvrcdeva": "Ever been divorced/had civil union dissolved",
                "edctn": "Doing last 7 days: education",
                "edctnp": "Partner doing last 7 days: education",
                "edufld": "Field or subject, highest qualification",
                "edulvla": "Highest level of education",
                "edulvlb": "Highest level of education",
                "edulvlfa": "Father's highest level of education",
                "edulvlfb": "Father's highest level of education",
                "edulvlma": "Mother's highest level of education",
                "edulvlmb": "Mother's highest level of education",
                "edulvlpa": "Partner's highest level of education",
                "edulvlpb": "Partner's highest level of education",
                "eduyrs": "Years of full-time education completed",
                "eisced": "Highest level of education, ES - ISCED",
                "eiscedf": "Father's highest level of education, ES - ISCED",
                "eiscedm": "Mother's highest level of education, ES - ISCED",
                "eiscedp": "Partner's highest level of education, ES - ISCED",
                "emplno": "Number of employees respondent has/had",
                "emplnof": "Number of employees father had",
                "emplnom": "Number of employees mother had",
                "emplnop": "Number of employees partner has",
                "emplrel": "Employment relation",
                "emprelp": "Partner's employment relation",
                "emprf14": "Father's employment status when respondent 14",
                "emprm14": "Mother's employment status when respondent 14",
                "estsz": "Establishment size",
                "hincfel": "Feeling about household's income nowadays",
                "hincsrca": "Main source of household income",
                "hinctnta": "Household's total net income, all sources",
                "hswrk": "Doing last 7 days: housework, looking after children, others",
                "hswrkp": "Partner doing last 7 days: housework, looking after children, others",
                "iorgact": "Allowed to influence policy decisions about activities of organisation",
                "isco08": "Occupation, ISCO08",
                "isco08p": "Occupation partner, ISCO08",
                "iscoco": "Occupation, ISCO88 (com)",
                "iscocop": "Occupation partner, ISCO88 (com)",
                "jbspv": "Responsible for supervising other employees",
                "jbspvf": "Father responsible for supervising other employees",
                "jbspvm": "Mother responsible for supervising other employees",
                "jbspvp": "Partner responsible for supervising other employees",
                "lvgptne": "Ever lived with a partner without being married",
                "lvgptnea": "Ever lived with a partner, without being married",
                "mainact": "Main activity last 7 days",
                "maritalb": "Legal marital status, post coded",
                "marsts": "Legal marital status",
                "mbtru": "Member of trade union or similar organisation",
                "mnactic": "Main activity, last 7 days. All respondents. Post coded",
                "mnactp": "Partner's main activity last 7 days",
                "njbspv": "Number of people responsible for in job",
                "njbspvp": "Number of people partner responsible for in job",
                "occf14b": "Father's occupation when respondent 14",
                "occm14b": "Mother's occupation when respondent 14",
                "partner": "Lives with husband/wife/partner at household grid",
                "pdjobev": "Ever had a paid job",
                "pdjobyr": "Year last in paid job",
                "pdwrk": "Doing last 7 days: paid work",
                "pdwrkp": "Partner doing last 7 days: paid work",
                "rshpsts": "Relationship with husband/wife/partner currently living with",
                "rtrd": "Doing last 7 days: retired",
                "rtrdp": "Partner doing last 7 days: retired",
                "tporgwk": "What type of organisation work/worked for",
                "uemp12m": "Any period of unemployment and work seeking lasted 12 months or more",
                "uemp3m": "Ever unemployed and seeking work for a period more than three months",
                "uemp5yr": "Any period of unemployment and work seeking within last 5 years",
                "uempla": "Doing last 7 days: unemployed, actively looking for job",
                "uemplap": "Partner doing last 7 days: unemployed, actively looking for job",
                "uempli": "Doing last 7 days: unemployed, not actively looking for job",
                "uemplip": "Partner doing last 7 days: unemployed, not actively looking for job",
                "wkhct": "Total contracted hours per week in main job overtime excluded",
                "wkhtot": "Total hours normally worked per week in main job overtime included",
                "wkhtotp": "Hours normally worked a week in main job overtime included, partner",
                "wrkac6m": "Paid work in another country, period more than 6 months last 10 years",
                "wrkctra": "Employment contract unlimited or limited duration",
                "wkdcorg": "Allowed to decide how daily work is organised",
                "wkdcorga": "Allowed to decide how daily work is organised",
                "nacer1": "Industry, NACE rev.1",
                "nacer11": "Industry, NACE rev.1.1",
                "nacer2": "Industry, NACE rev.2",
            }
        elif name == "chars":
            self.question_mapping = {
                "hhmmb": "Number of people living regularly as member of household",
                "gndr": "Gender",
                "gndr2": "Gender of second person in household",
                "gndr3": "Gender of third person in household",
                "gndr4": "Gender of fourth person in household",
                "gndr5": "Gender of fifth person in household",
                "gndr6": "Gender of sixth person in household",
                "gndr7": "Gender of seventh person in household",
                "gndr8": "Gender of eighth person in household",
                "gndr9": "Gender of ninth person in household",
                "gndr10": "Gender of tenth person in household",
                "gndr11": "Gender of eleventh person in household",
                "gndr12": "Gender of twelfth person in household",
                "gndr13": "Gender of thirteenth person in household",
                "gndr14": "Gender of fourteenth person in household",
                "gndr15": "Gender of fifteenth person in household",
                "rshipa2": "Second person in household: relationship to respondent",
                "rshipa3": "Third person in household: relationship to respondent",
                "rshipa4": "Fourth person in household: relationship to respondent",
                "rshipa5": "Fifth person in household: relationship to respondent",
                "rshipa6": "Sixth person in household: relationship to respondent",
                "rshipa7": "Seventh person in household: relationship to respondent",
                "rshipa8": "Eighth person in household: relationship to respondent",
                "rshipa9": "Ninth person in household: relationship to respondent",
                "rshipa10": "Tenth person in household: relationship to respondent",
                "rshipa11": "Eleventh person in household: relationship to respondent",
                "rshipa12": "Twelfth person in household: relationship to respondent",
                "rshipa13": "Thirteenth person in household: relationship to respondent",
                "rshipa14": "Fourteenth person in household: relationship to respondent",
                "rshipa15": "Fifteenth person in household: relationship to respondent",
                "yrbrn": "Year of birth",
                "yrbrn2": "Year of birth of second person in household",
                "yrbrn3": "Year of birth of third person in household",
                "yrbrn4": "Year of birth of fourth person in household",
                "yrbrn5": "Year of birth of fifth person in household",
                "yrbrn6": "Year of birth of sixth person in household",
                "yrbrn7": "Year of birth of seventh person in household",
                "yrbrn8": "Year of birth of eighth person in household",
                "yrbrn9": "Year of birth of ninth person in household",
                "yrbrn10": "Year of birth of tenth person in household",
                "yrbrn11": "Year of birth of eleventh person in household",
                "yrbrn12": "Year of birth of twelfth person in household",
                "yrbrn13": "Year of birth of thirteenth person in household",
                "yrbrn14": "Year of birth of fourteenth person in household",
                "yrbrn15": "Year of birth of fifteenth person in household",
                "agea": "Age of respondent, calculated"
            }
        elif name == "wellbeing":
            self.question_mapping = {
                "aesfdrk": "Feeling of safety of walking alone in local area after dark",
                "atchctr": "How emotionally attached to [country]",
                "atcherp": "How emotionally attached to Europe",
                "blgetmg": "Belong to minority ethnic group in country",
                "brncntr": "Born in country",
                "cntbrth": "Country of birth",
                "cntbrtha": "Country of birth",
                "cntbrthb": "Country of birth",
                "cntbrthc": "Country of birth",
                "cntbrthd": "Country of birth",
                "crmvct": "Respondent or household member victim of burglary/assault last 5 years",
                "ctzcntr": "Citizen of country",
                "ctzship": "Citizenship",
                "ctzshipa": "Citizenship",
                "ctzshipb": "Citizenship",
                "ctzshipc": "Citizenship",
                "ctzshipd": "Citizenship",
                "dscrage": "Discrimination of respondent's group: age",
                "dscrdk": "Discrimination of respondent's group: don't know",
                "dscrdsb": "Discrimination of respondent's group: disability",
                "dscretn": "Discrimination of respondent's group: ethnic group",
                "dscrgnd": "Discrimination of respondent's group: gender",
                "dscrgrp": "Member of a group discriminated against in this country",
                "dscrlng": "Discrimination of respondent's group: language",
                "dscrna": "Discrimination of respondent's group: no answer",
                "dscrnap": "Discrimination of respondent's group: not applicable",
                "dscrntn": "Discrimination of respondent's group: nationality",
                "dscroth": "Discrimination of respondent's group: other grounds",
                "dscrrce": "Discrimination of respondent's group: colour or race",
                "dscrref": "Discrimination of respondent's group: refusal",
                "dscrrlg": "Discrimination of respondent's group: religion",
                "dscrsex": "Discrimination of respondent's group: sexuality",
                "facntr": "Father born in country",
                "fbrncnt": "Country of birth, father",
                "fbrncnta": "Country of birth, father",
                "fbrncntb": "Country of birth, father",
                "fbrncntc": "Country of birth, father",
                "happy": "How happy are you",
                "health": "Subjective general health",
                "hlthhmp": "Hampered in daily activities by illness/disability/infirmity/mental problem",
                "inmdisc": "Anyone to discuss intimate and personal matters with",
                "inprdsc": "How many people with whom you can discuss intimate and personal matters",
                "livecntr": "How long ago first came to live in country",
                "livecnta": "What year you first came to live in country",
                "lnghoma": "Language most often spoken at home: first mentioned",
                "lnghom1": "Language most often spoken at home: first mentioned",
                "lnghomb": "Language most often spoken at home: second mentioned",
                "lnghom2": "Language most often spoken at home: second mentioned",
                "mbrncnt": "Country of birth, mother",
                "mbrncnta": "Country of birth, mother",
                "mbrncntb": "Country of birth, mother",
                "mbrncntc": "Country of birth, mother",
                "mocntr": "Mother born in country",
                "pray": "How often pray apart from at religious services",
                "rlgatnd": "How often attend religious services apart from special occasions",
                "rlgblg": "Belonging to particular religion or denomination",
                "rlgblge": "Ever belonging to particular religion or denomination",
                "rlgdgr": "How religious are you",
                "rlgdnm": "Religion or denomination belonging to at present",
                "rlgdnme": "Religion or denomination belonging to in the past",
                "sclact": "Take part in social activities compared to others of same age",
                "sclmeet": "How often socially meet with friends, relatives or colleagues",
                "ccnthum": "Climate change caused by natural processes, human activity, or both",
                "ccrdprs": "To what extent feel personal responsibility to reduce climate change",
                "wrclmch": "How worried about climate change",
                "vteurmmb": "Would vote for [country] to remain member of European Union or leave",
                "vteubcmb": "Would vote for [country] to become member of European Union or remain outside"
            }
        elif name == "media":
            self.question_mapping = {
                "netuse": "Personal use of internet/e-mail/www",
                "netusoft": "Internet use, how often",
                "netustm": "Internet use, how much time on typical day, in minutes",
                "nwspol": "News about politics and current affairs, watching, reading or listening, in minutes",
                "nwsppol": "Newspaper reading, politics/current affairs on average weekday",
                "nwsptot": "Newspaper reading, total time on average weekday",
                "pplfair": "Most people try to take advantage of you, or try to be fair",
                "pplhlp": "Most of the time people helpful or mostly looking out for themselves",
                "ppltrst": "Most people can be trusted or you can't be too careful",
                "rdpol": "Radio listening, news/politics/current affairs on average weekday",
                "rdtot": "Radio listening, total time on average weekday",
                "tvpol": "TV watching, news/politics/current affairs on average weekday",
                "tvtot": "TV watching, total time on average weekday"
            }

    def clean_special_values(self):
        """Clean special values from the dataframe"""
        for col in self.df.columns:
            if col in ['cntry', 'essround']:  # Skip identifier columns
                continue

            # Get column data type and values
            dtype = self.df[col].dtype
            unique_vals = self.df[col].unique()

            # Create masks for different special values based on column characteristics
            if dtype == 'int64' or dtype == 'float64':
                # Check column range to determine the pattern length
                if len(unique_vals) > 0:
                    max_val = max([x for x in unique_vals if isinstance(x, (int, float)) and not np.isnan(x)], default=0)

                    # For year-type variables (4-digit numbers)
                    if max_val > 1000:
                        self.df[col] = self.df[col].replace([6666, 7777, 8888, 9999], np.nan)
                    # For 1-10 scale questions
                    elif max_val > 10 and max_val < 100:
                        self.df[col] = self.df[col].replace([66, 77, 88, 99], np.nan)
                    # For standard questions with smaller scales
                    else:
                        self.df[col] = self.df[col].replace([6, 7, 8, 9], np.nan)

    @property
    def questions(self):
        """Lazy-loaded question data dictionary with mode values"""
        # This creates the appearance of the questions dictionary without pre-computing all values
        return LazyQuestionDict(self)

    @property
    def questionsMean(self):
        """Lazy-loaded question data dictionary with mean values"""
        # This creates the appearance of the questionsMean dictionary without pre-computing all values
        return LazyQuestionMeanDict(self)

    def get_question_data(self, question, mode_or_mean='mode'):
        """Get data for a specific question, computing it if not cached"""
        cache = self._question_cache if mode_or_mean == 'mode' else self._mean_cache

        # Return cached result if available
        if question in cache:
            return cache[question]

        # Get the original column name if this is a mapped question
        original_col = self._get_original_column(question)

        # Process the question
        print(f"Processing question: {question}")
        start_time = time.time()

        results = {}
        for country, country_data in self.country_data_dict.items():
            country_results = ["None"] * 11

            round_groups = country_data.groupby("essround")
            for round_num, round_data in round_groups:
                if 1 <= round_num <= 11 and not round_data[original_col].isna().all():
                    round_data_filtered = round_data[original_col].dropna()

                    if not round_data_filtered.empty:
                        if mode_or_mean == 'mode':
                            # Calculate most common (mode)
                            most_common = round_data_filtered.value_counts().idxmax()
                            country_results[round_num - 1] = most_common
                        else:  # mean
                            # Calculate mean if the data is numeric
                            try:
                                numeric_data = pd.to_numeric(round_data_filtered, errors='coerce')
                                if not numeric_data.isna().all():
                                    mean_value = numeric_data.mean()
                                    country_results[round_num - 1] = round(mean_value, 2)
                            except:
                                pass  # Keep as "None" if mean can't be calculated

            results[country] = country_results

        # Cache the result
        cache[question] = results

        end_time = time.time()
        print(f"Processed {question} in {end_time - start_time:.4f} seconds")

        return results

    def _get_original_column(self, question):
        """Get the original column name for a mapped question"""
        original_col = question
        if self.question_mapping:
            # Find the original column name by looking up the description
            for col, desc in self.question_mapping.items():
                if desc == question:
                    original_col = col
                    break
        return original_col


class LazyQuestionDict:
    """A dictionary-like object that loads question data on demand"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, question):
        return self.dataset.get_question_data(question, 'mode')

    def keys(self):
        return self.dataset.questionLabels


class LazyQuestionMeanDict:
    """A dictionary-like object that loads question mean data on demand"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, question):
        return self.dataset.get_question_data(question, 'mean')

    def keys(self):
        return self.dataset.questionLabels


def main():
    start_time = time.time()

    politics = Dataset("politics")
    print("Dataset loaded, now retrieving a sample question")

    # This will trigger the lazy loading of just one question
    first_question = politics.questionLabels[0]
    first_country = politics.countryLabels[0]
    data = politics.questions[first_question]

    print(f"{first_question} for {first_country}: {data[first_country]}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()