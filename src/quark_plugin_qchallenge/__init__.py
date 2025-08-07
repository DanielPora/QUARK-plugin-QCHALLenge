from quark.plugin_manager import factory

from quark_plugin_qchallenge.sp_qubo_mapping import SpQuboMapping
from quark_plugin_qchallenge.sp_problem_provider import SpProblemProvider


def register() -> None:
    factory.register("qch_sp_qubo_maping", SpQuboMapping)
    factory.register("qch_sp_problem_provider", SpProblemProvider)
