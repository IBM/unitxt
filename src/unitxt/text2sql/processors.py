import re

from ..operators import FieldOperator


class AddPrefix(FieldOperator):
    prefix: str

    def process_value(self, text: str) -> str:
        text = text.strip()
        if text.startswith(self.prefix):
            return text
        return self.prefix + text.strip()


class GetSQL(FieldOperator):
    def process_value(self, text: str) -> str:
        """Extracts the first SQL query from a given text.

        Args:
        text: The input string containing the SQL query.

        Returns:
        The first SQL query found in the text, or None if no query is found.
        """
        match = re.search(
            r"(?:```)?.*?((?:.*?)SELECT(?:.*?))(?:```|;|$)", text, re.DOTALL
        )

        if match:
            out = text[match.start() : match.end()].replace("```", "").replace(";", "")
        else:
            out = "No query found in generation"

        return out

    # def process_value(self, text: str) -> str:
    #     text = text.strip()
    #     if "\n\n" in text:
    #         text = text.split("\n\n")[0]
    #     if "<|eot_id|>" in text:
    #         text = text[: text.find("<|eot_id|>")]
    #     if "SELECT" in text and ";" in text:
    #         return text[text.find("SELECT") : text.find(";") + 1]
    #     if "SELECT" in text:
    #         return text[text.find("SELECT") :]
    #     return text


# class StripCodeBlock(FieldOperator):
#     def process_value(self, text: str) -> str:
#         text = text.strip()
#         if text.startswith("```sql"):
#             text = text.replace("```sql", "", 1)
#         if text.endswith("```"):
#             text = text.replace("```", "", 1)
#         if "```" in text:
#             text = text.split("```")[0]
#         return text


# raw prediction
# text = "You are a Text2SQL generation model, in your answer, only have SQL code\nYou are given the following SQL schema\n\n```sql\nCREATE TABLE frpm\n(\n    CDSCode                                       TEXT not null\n        primary key,\n    `Academic Year`                               TEXT  null,\n    `County Code`                                 TEXT  null,\n    `District Code`                               INTEGER         null,\n    `School Code`                                 TEXT  null,\n    `County Name`                                 TEXT null,\n    `District Name`                               TEXT null,\n    `School Name`                                 TEXT null,\n    `District Type`                               TEXT null,\n    `School Type`                                 TEXT null,\n    `Educational Option Type`                     TEXT null,\n    `NSLP Provision Status`                       TEXT null,\n    `Charter School (Y/N)`                        INTEGER    null,\n    `Charter School Number`                       TEXT  null,\n    `Charter Funding Type`                        TEXT null,\n    IRC                                           INTEGER    null,\n    `Low Grade`                                   TEXT  null,\n    `High Grade`                                  TEXT null,\n    `Enrollment (K-12)`                           REAL      null,\n    `Free Meal Count (K-12)`                      REAL       null,\n    `Percent (%) Eligible Free (K-12)`            REAL       null,\n    `FRPM Count (K-12)`                           REAL       null,\n    `Percent (%) Eligible FRPM (K-12)`            REAL       null,\n    `Enrollment (Ages 5-17)`                      REAL       null,\n    `Free Meal Count (Ages 5-17)`                 REAL       null,\n    `Percent (%) Eligible Free (Ages 5-17)`       REAL       null,\n    `FRPM Count (Ages 5-17)`                      REAL       null,\n    `Percent (%) Eligible FRPM (Ages 5-17)`       REAL       null,\n    `2013-14 CALPADS Fall 1 Certification Status` INTEGER    null,\n    foreign key (CDSCode) references schools (CDSCode)\n)\n\nCREATE TABLE satscores\n(\n    cds         TEXT not null\n        primary key,\n    rtype       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    enroll12    INTEGER         not null,\n    NumTstTakr  INTEGER          not null,\n    AvgScrRead  INTEGER          null,\n    AvgScrMath  INTEGER          null,\n    AvgScrWrite INTEGER          null,\n    NumGE1500   INTEGER          null,\n--     PctGE1500   double      null,\n        foreign key (cds) references schools (CDSCode)\n)\n\nCREATE TABLE schools\n(\n    CDSCode     TEXT not null\n        primary key,\n    NCESDist    TEXT  null,\n    NCESSchool  TEXT  null,\n    StatusType  TEXT  not null,\n    County      TEXT not null,\n    District    TEXT not null,\n    School      TEXT null,\n    Street      TEXT null,\n    StreetAbr   TEXT null,\n    City        TEXT null,\n    Zip         TEXT null,\n    State       TEXT  null,\n    MailStreet  TEXT null,\n    MailStrAbr  TEXT null,\n    MailCity    TEXT null,\n    MailZip     TEXT null,\n    MailState   TEXT  null,\n    Phone       TEXT null,\n    Ext         TEXT  null,\n    Website     TEXT null,\n    OpenDate    DATE        null,\n    ClosedDate  DATE        null,\n    Charter     INTEGER    null,\n    CharterNum  TEXT  null,\n    FundingType TEXT null,\n    DOC         TEXT  not null,\n    DOCType     TEXT not null,\n    SOC         TEXT  null,\n    SOCType     TEXT null,\n    EdOpsCode   TEXT  null,\n    EdOpsName   TEXT null,\n    EILCode     TEXT  null,\n    EILName     TEXT null,\n    GSoffered   TEXT null,\n    GSserved    TEXT  null,\n    Virtual     TEXT  null,\n    Magnet      INTEGER   null,\n    Latitude    REAL      null,\n    Longitude   REAL      null,\n    AdmFName1   TEXT null,\n    AdmLName1   TEXT null,\n    AdmEmail1   TEXT null,\n    AdmFName2   TEXT null,\n    AdmLName2   TEXT null,\n    AdmEmail2   TEXT null,\n    AdmFName3   TEXT  null,\n    AdmLName3   TEXT null,\n    AdmEmail3   TEXT null,\n    LastUpdate  DATE        not null\n)```\n\nand question:\n\nWhat is the highest eligible free rate for K-12 students in the schools in Alameda County?\n```sql\nSELECT  T1.Percent (%) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Alameda County' AND T1.NCESDist = 'K-12' ORDER BY T1.Percent (%) DESC LIMIT 1\n```\n\nThis query selects the highest eligible free rate for K-12 students in the schools in Alameda County. It uses the percentages of the eligible free rate for K-12 students in the schools in Alameda County. It does this by selecting the eligible free rate for K-12 students in the schools in Alameda County and then using the ORDER BY clause to order the results by the percentage of eligible free rate for K-12 students in the schools in Alameda County. The result is the percentage of eligible free rate for K-12 students in the schools in Alameda County.\n\nThe SQL query selects the eligible free rate for K-12 students in the schools in Alameda County. It uses the percentages of the eligible free rate for K-"


# #### processed prediction
# "SELECT You are a Text2SQL generation model, in your answer, only have SQL code\nYou are given the following SQL schema"


# def extract_sql_query(text):
#     """Extracts the first SQL query from a given text.

#     Args:
#       text: The input string containing the SQL query.

#     Returns:
#       The first SQL query found in the text, or None if no query is found.
#     """
#     match = re.search(r"```.*?SELECT.*?```", text, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     else:
#         return ""


# extract_sql_query()


# match = re.search(r"```.*?SELECT.*?```", text, re.DOTALL)
# text[match.start() : match.end()]

text = "```\nSELECT MAX(`Percent (%) Eligible Free (K-12)`) \nFROM frpm \nWHERE `County Name` = 'Alameda County';\n```"
text = "SELECT s.Zip \nFROM frpm f \nJOIN schools s ON f.CDSCode = s.CDSCode \nWHERE f.`Charter School (Y/N)` = 1 AND f.`County Name` = 'Fresno County Office of Education';"

match = re.search(r"(?:```)?.*?((?:.*?)SELECT(?:.*?))(?:```|;|$)", text, re.DOTALL)

text[match.start() : match.end()].replace("```", "").replace(";", "")
