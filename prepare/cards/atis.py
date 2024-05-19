from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, TaskCard
from unitxt.operators import (
    AddFields,
    CopyFields,
)
from unitxt.processors import SplitStrip
from unitxt.span_lableing_operators import IobExtractor
from unitxt.test_utils.card import test_card

classes = [
    "aircraft_code",
    "airline_code",
    "airline_name",
    "airport_code",
    "airport_name",
    "arrive_date.date_relative",
    "arrive_date.day_name",
    "arrive_date.day_number",
    "arrive_date.month_name",
    "arrive_date.today_relative",
    "arrive_time.end_time",
    "arrive_time.period_mod",
    "arrive_time.period_of_day",
    "arrive_time.start_time",
    "arrive_time.time",
    "arrive_time.time_relative",
    "city_name",
    "class_type",
    "connect",
    "cost_relative",
    "day_name",
    "day_number",
    "days_code",
    "depart_date.date_relative",
    "depart_date.day_name",
    "depart_date.day_number",
    "depart_date.month_name",
    "depart_date.today_relative",
    "depart_date.year",
    "depart_time.end_time",
    "depart_time.period_mod",
    "depart_time.period_of_day",
    "depart_time.start_time",
    "depart_time.time",
    "depart_time.time_relative",
    "economy",
    "fare_amount",
    "fare_basis_code",
    "flight_days",
    "flight_mod",
    "flight_number",
    "flight_stop",
    "flight_time",
    "fromloc.airport_code",
    "fromloc.airport_name",
    "fromloc.city_name",
    "fromloc.state_code",
    "fromloc.state_name",
    "meal",
    "meal_code",
    "meal_description",
    "mod",
    "month_name",
    "or",
    "period_of_day",
    "restriction_code",
    "return_date.date_relative",
    "return_date.day_name",
    "return_date.day_number",
    "return_date.month_name",
    "return_date.today_relative",
    "return_time.period_mod",
    "return_time.period_of_day",
    "round_trip",
    "state_code",
    "state_name",
    "stoploc.airport_name",
    "stoploc.city_name",
    "stoploc.state_code",
    "time",
    "time_relative",
    "today_relative",
    "toloc.airport_code",
    "toloc.airport_name",
    "toloc.city_name",
    "toloc.country_name",
    "toloc.state_code",
    "toloc.state_name",
    "transport_type",
]

card = TaskCard(
    loader=LoadHF(
        path="tuetschek/atis",
    ),
    preprocess_steps=[
        SplitStrip(
            delimiter=" ",
            field_to_field={
                "slots": "labels",
                "text": "tokens",
            },
        ),
        IobExtractor(
            labels=classes,
            begin_labels=["B-" + c for c in classes],
            inside_labels=["I-" + c for c in classes],
            outside_label="O",
        ),
        CopyFields(
            field_to_field={
                "spans/*/start": "spans_starts",
                "spans/*/end": "spans_ends",
                "spans/*/label": "labels",
            },
            get_default=[],
            not_exist_ok=True,
        ),
        AddFields(
            fields={
                "text_type": "text",
                "class_type": "entity type",
                "classes": classes,
            }
        ),
    ],
    task="tasks.span_labeling.extraction",
    templates="templates.span_labeling.extraction.all",
    __tags__={"croissant": True, "region": "us"},
)

test_card(card)

add_to_catalog(card, "cards.atis", overwrite=True)
