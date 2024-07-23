from unitxt.blocks import Copy, LoadHF, Set, SplitRandomMix, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

# https://localizely.com/iso-639-2-list/
iso_lang_code_mapping = {
    "eng": "English",
    "afr": "Afrikaans",
    "amh": "Amharic",
    "ara": "Arabic",
    "hye": "Armenian",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "mya": "Burmese",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "zho_simpl": "Chinese (Simplified)",
    "zho_trad": "Chinese (Traditional)",
    "hrv": "Croatian",
    "ces": "Czech",
    "dan": "Danish",
    "nld": "Dutch",
    "est": "Estonian",
    "tgl": "Tagalog",
    "fin": "Finnish",
    "fra": "French",
    "ful": "Fulah",
    "glg": "Galician",
    "lug": "Ganda",
    "kat": "Georgian",
    "deu": "German",
    "ell": "Greek",
    "guj": "Gujarati",
    "hau": "Hausa",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hun": "Hungarian",
    "isl": "Icelandic",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "gle": "Irish",
    "ita": "Italian",
    "jpn": "Japanese",
    "jav": "Javanese",
    "kea": "Kabuverdianu",
    "kam": "Kamba",
    "kan": "Kannada",
    "kaz": "Kazakh",
    "khm": "Khmer",
    "kor": "Korean",
    "kir": "Kyrgyz",
    "lao": "Lao",
    "lav": "Latvian",
    "lin": "Lingala",
    "lit": "Lithuanian",
    "luo": "Dholuo",
    "ltz": "Luxembourgish",
    "mkd": "Macedonian",
    "msa": "Malay",
    "mal": "Malayalam",
    "mlt": "Maltese",
    "mri": "Maori",
    "mar": "Marathi",
    "mon": "Mongolian",
    "npi": "Nepali",
    "nso": "Northern Sotho",
    "nob": "Norwegian Bokmål",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "orm": "Oromo",
    "pus": "Pashto",
    "fas": "Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "pan": "Punjabi",
    "ron": "Romanian",
    "rus": "Russian",
    "srp": "Serbian",
    "sna": "Shona",
    "snd": "Sindhi",
    "slk": "Slovak",
    "slv": "Slovenian",
    "som": "Somali",
    "ckb": "Sorani Kurdish",
    "spa": "Spanish",
    "swh": "Swahili",
    "swe": "Swedish",
    "tgk": "Tajik",
    "tam": "Tamil",
    "tel": "Telugu",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "umb": "Umbundu",
    "urd": "Urdu",
    "uzb": "Uzbek",
    "vie": "Vietnamese",
    "cym": "Welsh",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
}


langs_to_include = [  # langs currently supported by sacrebleu
    "ara",
    "fra",
    "deu",
    "jpn",
    "kor",
    "por",
    "ron",
    "spa",
]

langs = [
    lang
    for lang in iso_lang_code_mapping.keys()
    if ("eng" not in lang and lang in langs_to_include)
]
pairs = [{"src": lang, "tgt": "eng"} for lang in langs] + [
    {"src": "eng", "tgt": lang} for lang in langs
]

for pair in pairs:
    card = TaskCard(
        loader=LoadHF(path="gsarti/flores_101", name="all"),
        preprocess_steps=[
            SplitRandomMix({"validation": "dev", "test": "devtest"}),
            Copy(
                field_to_field={
                    f"sentence_{pair['src']}": "text",
                    f"sentence_{pair['tgt']}": "translation",
                },
            ),
            Set(
                fields={
                    "source_language": iso_lang_code_mapping[pair["src"]].lower(),
                    "target_language": iso_lang_code_mapping[pair["tgt"]].lower(),
                }
            ),
        ],
        task="tasks.translation.directed",
        templates="templates.translation.directed.all",
    )

    test_card(card, demos_taken_from="test")
    add_to_catalog(
        card, f"cards.mt.flores_101.{pair['src']}_{pair['tgt']}", overwrite=True
    )

if __name__ == "__main__":
    from unitxt import load_dataset

    ds = load_dataset(
        "card=cards.mt.flores_101.eng_deu,template_card_index=0",
    )

    ds["test"][0]
