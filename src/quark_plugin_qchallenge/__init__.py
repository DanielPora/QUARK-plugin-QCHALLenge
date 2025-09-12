from quark.plugin_manager import factory

from quark_plugin_qchallenge.qch_sensor_positioning import SPProblem
from quark_plugin_qchallenge.qch_auto_carrier_loading import ACLProblem
from quark_plugin_qchallenge.qch_sensor_positioning_qubo import SPProblemQubo


def register() -> None:
    factory.register("qch_sensor_positioning", SPProblem)
    factory.register("qch_sensor_positioning_qubo", SPProblemQubo)
    factory.register("qch_auto_carrier_loading", ACLProblem)
