from quark.plugin_manager import factory

from quark_plugin_qchallenge.qch_sensor_positioning_toy import SpToyProblem
from quark_plugin_qchallenge.qch_sensor_positioning_real import SpRealProblem
from quark_plugin_qchallenge.qch_sensor_positioning_qubo_toy import SpToyProblemQubo


def register() -> None:
    factory.register("qch_sensor_positioning_toy", SpToyProblem)
    factory.register("qch_sensor_positioning_real", SpRealProblem)
    factory.register("qch_sensor_positioning_toy_qubo", SpToyProblemQubo)
