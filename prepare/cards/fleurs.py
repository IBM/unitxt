# This Python script is used to prepare cards for the fleurs dataset, used for evaluating speech translation

from unitxt.audio_operators import ToAudio
from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import MultipleSourceLoader
from unitxt.operator import SourceSequentialOperator
from unitxt.operators import AddFields, RemoveFields, Rename
from unitxt.splitters import RenameSplits
from unitxt.stream_operators import JoinStreamsFleurs
from unitxt.test_utils.card import test_card

# Entire list of languages (codes) supported by fleurs, for reference:
# all_subsets = [
#     "af_za",
#     "am_et",
#     "ar_eg",
#     "as_in",
#     "ast_es",
#     "az_az",
#     "be_by",
#     "bg_bg",
#     "bn_in",
#     "bs_ba",
#     "ca_es",
#     "ceb_ph",
#     "ckb_iq",
#     "cmn_hans_cn",
#     "cs_cz",
#     "cy_gb",
#     "da_dk",
#     "de_de",
#     "el_gr",
#     "en_us",
#     "es_419",
#     "et_ee",
#     "fa_ir",
#     "ff_sn",
#     "fi_fi",
#     "fil_ph",
#     "fr_fr",
#     "ga_ie",
#     "gl_es",
#     "gu_in",
#     "ha_ng",
#     "he_il",
#     "hi_in",
#     "hr_hr",
#     "hu_hu",
#     "hy_am",
#     "id_id",
#     "ig_ng",
#     "is_is",
#     "it_it",
#     "ja_jp",
#     "jv_id",
#     "ka_ge",
#     "kam_ke",
#     "kea_cv",
#     "kk_kz",
#     "km_kh",
#     "kn_in",
#     "ko_kr",
#     "ky_kg",
#     "lb_lu",
#     "lg_ug",
#     "ln_cd",
#     "lo_la",
#     "lt_lt",
#     "luo_ke",
#     "lv_lv",
#     "mi_nz",
#     "mk_mk",
#     "ml_in",
#     "mn_mn",
#     "mr_in",
#     "ms_my",
#     "mt_mt",
#     "my_mm",
#     "nb_no",
#     "ne_np",
#     "nl_nl",
#     "nso_za",
#     "ny_mw",
#     "oc_fr",
#     "om_et",
#     "or_in",
#     "pa_in",
#     "pl_pl",
#     "ps_af",
#     "pt_br",
#     "ro_ro",
#     "ru_ru",
#     "sd_in",
#     "sk_sk",
#     "sl_si",
#     "sn_zw",
#     "so_so",
#     "sr_rs",
#     "sv_se",
#     "sw_ke",
#     "ta_in",
#     "te_in",
#     "tg_tj",
#     "th_th",
#     "tr_tr",
#     "uk_ua",
#     "umb_ao",
#     "ur_pk",
#     "uz_uz",
#     "vi_vn",
#     "wo_sn",
#     "xh_za",
#     "yo_ng",
#     "yue_hant_hk",
#     "zu_za",
# ]

# Currently we only use fleurs for evaluating translating from english_us to the following 6 languages; additions may follow
# The fleurs dataset can be used from any language to any language within its supported list of languages
# We use the (basic) Whisper text normalization; before extending to other languages, check if Whisper basic normalizer supports them

# "to" language:
target_subsets = [
    "de_de",
    "es_419",
    "fr_fr",
    "it_it",
    "ja_jp",
    "pt_br",
    "cmn_hans_cn",
]

# "from" language:
source_subsets = [
    "en_us",
]

lang_code = {
    "en_us": "en",
    "de_de": "de",
    "es_419": "es",
    "fr_fr": "fr",
    "it_it": "it",
    "ja_jp": "ja",
    "pt_br": "pt",
    "cmn_hans_cn": "zh",
}

lang_name = {
    "en_us": "English",
    "de_de": "German",
    "es_419": "Spanish",
    "fr_fr": "French",
    "it_it": "Italian",
    "ja_jp": "Japanese",
    "pt_br": "Portuguese",
    "cmn_hans_cn": "Chinese",
}

first = True
for source_subset in source_subsets:
    for target_subset in target_subsets:
        card = TaskCard(
            loader=MultipleSourceLoader(
                sources=[
                    SourceSequentialOperator(
                        steps=[
                            LoadHF(
                                path="google/fleurs",
                                revision="refs/convert/parquet",
                                data_dir=source_subset,
                                splits=["test"],
                                data_classification_policy=["public"],
                            ),
                            RemoveFields(
                                fields=[
                                    "num_samples",
                                    "path",
                                    "raw_transcription",
                                    "transcription",
                                    "gender",
                                    "lang_id",
                                    "language",
                                    "lang_group_id",
                                ]
                            ),
                            RenameSplits(
                                {
                                    "test": "test_input",
                                }
                            ),
                        ]
                    ),
                    SourceSequentialOperator(
                        steps=[
                            LoadHF(
                                path="google/fleurs",
                                revision="refs/convert/parquet",
                                data_dir=target_subset,
                                splits=["test"],
                                data_classification_policy=["public"],
                            ),
                            RemoveFields(
                                fields=[
                                    "num_samples",
                                    "path",
                                    "audio",
                                    "raw_transcription",
                                    "gender",
                                    "lang_id",
                                    "language",
                                    "lang_group_id",
                                ]
                            ),
                            RenameSplits(
                                {
                                    "test": "test_target",
                                }
                            ),
                        ]
                    ),
                ]
            ),
            preprocess_steps=[
                # The following JointStream operator is unique for fleurs
                # It removes redundant repetitions in the target ("right") stream - repetitions of the same translation text targets
                # The repetitions are inherent to the structure of the fleurs dataset
                # On the other hand, we do preserve the repeated recornings of the input speech (1-3 recordings per target instance),
                #    as they contain recordings by different speakers and different recording conditions, therefore considered as
                #    separate instannces
                # The unique JointStream operator also adds '.' at the end of text instances, to improve the behavior of the metric (sacrebleu)
                JoinStreamsFleurs(
                    left_stream="test_input",
                    right_stream="test_target",
                    how="inner",
                    on=["id"],
                    new_stream_name="test",
                ),
                ToAudio(field="audio"),
                Rename(field="transcription", to_field="translation"),
                AddFields(
                    {
                        "source_language": lang_name[source_subset],
                        "target_language": lang_name[target_subset],
                    }
                ),
            ],
            task="tasks.translation.speech",
            templates=[
                "templates.translation.speech.default",
            ],
        )
        if first:
            test_card(card, demos_taken_from="test", num_demos=0)
            first = False
        add_to_catalog(
            card, f"cards.fleurs.{source_subset}.{target_subset}", overwrite=True
        )
