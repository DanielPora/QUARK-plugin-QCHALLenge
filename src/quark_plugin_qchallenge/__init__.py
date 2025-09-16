from quark.plugin_manager import factory

from quark_plugin_qchallenge.qch_sensor_positioning import SPProblem
from quark_plugin_qchallenge.qch_sensor_positioning_qubo import SPProblemQubo
from quark_plugin_qchallenge.qch_auto_carrier_loading import ACLProblem
from quark_plugin_qchallenge.qch_production_assignment import PASProblem
from quark_plugin_qchallenge.qch_production_assignment_qubo import PASProblemQubo
from quark_plugin_qchallenge.qch_train_routing import TRProblem
from quark_plugin_qchallenge.qch_truck_loading import TLProblem


def register() -> None:
    factory.register("qch_sensor_positioning", SPProblem)
    factory.register("qch_sensor_positioning_qubo", SPProblemQubo)
    factory.register("qch_auto_carrier_loading", ACLProblem)
    factory.register("qch_production_assignment", PASProblem)
    factory.register("qch_production_assignment_qubo", PASProblemQubo)
    factory.register("qch_train_routing", TRProblem)
    factory.register("qch_truck_loading", TLProblem)