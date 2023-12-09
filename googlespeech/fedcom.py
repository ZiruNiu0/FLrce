from typing import List, Tuple, Union
from models import CNN
from fedcom_client import fedcom_client
import flwr as fl
import random
from flwr.common import Metrics
from flwr.common import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy.aggregate import weighted_loss_avg, aggregate
from flwr.common.logger import log
from logging import WARNING
from dataset import voice_dataset
from util import get_filters, get_parameters, set_filters, compute_sum, parameters_to_ndarrays, top_k_sparsification
CHANNEL = 1
Batch = 16
CLASSES = 35

class fedcom_strategy(fl.server.strategy.FedAvg):
    def __init__(self, ff, fe, mfc, mec, mac, ACC=[], ClientsSelection=[]):
        super().__init__(fraction_fit=ff, fraction_evaluate=fe, min_fit_clients=mfc, min_evaluate_clients=mec, min_available_clients=mac, evaluate_metrics_aggregation_fn=weighted_average)
        self.fraction_fit_=ff,
        self.fraction_evaluate_=fe,
        self.min_fit_clients_=mfc,
        self.min_evaluate_clients_=mec,
        self.min_available_clients_=mac
        self.global_model = CNN(CHANNEL, outputs=CLASSES)
        self.accuracy_record = ACC
        self.ClientMasks = {}
        self.local_residuals = {}
        self.local_models = {}
        self.droprate = 0.6
        self.latest_local_update = None
        for i in range(mac):
            self.local_residuals[i] = None

    def record_test_accuracy(self, acc):
        self.accuracy_record.append(acc)

    """override"""
    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters(get_parameters(self.global_model))
    
    """override"""
    def configure_fit(self, server_round: int, parameters, client_manager: ClientManager):
        random.seed(server_round)
        sample_size, min_num_clients = super().num_fit_clients(client_manager.num_available()) 
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_fit_list = []
        for client in clients:
            cid = int(client.cid)
            config = {}
            config['Residual'] = self.local_residuals[cid]
            sub_parameters = get_filters(self.global_model)
            fit_ins = FitIns(ndarrays_to_parameters(sub_parameters), config)
            config_fit_list.append((client, fit_ins))
        return config_fit_list
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
      """override"""
      """Aggregate fit results using weighted average."""
      if not results:
        return None, {}
      # Do not aggregate if there are failures and failures are not accepted
      if not self.accept_failures and failures:
        return None, {}
      # Convert results
      Fit_res = []
      for client, fit_res in results:
        cid = client.cid
        param, num = parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples
        Fit_res.append((param, 1))
        self.local_residuals[int(cid)] = fit_res.metrics["Residual"]
        self.local_models[int(cid)] = fit_res.metrics["personal model"]
      #for params, size, rate in weights_results:
      aggregated_updates = aggregate(Fit_res)
      current_global_model = get_filters(self.global_model)
      self.latest_local_update = top_k_sparsification(self.droprate, aggregated_updates)
      new_model = compute_sum(current_global_model, aggregated_updates)
      metrics_aggregated = {}
      if self.fit_metrics_aggregation_fn:
          fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
          metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
      elif server_round == 1:  # Only log this warning once
          log(WARNING, "No fit_metrics_aggregation_fn provided")
      set_filters(self.global_model, new_model)
      return new_model, metrics_aggregated
    
    def configure_evaluate(self, server_round: int, parameters, client_manager: ClientManager):
        """override"""
        if self.fraction_evaluate_ == 0.0:
            return []
        # Sample clients
        sample_size, min_num_clients = super().num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_evaluate_list = []
        parameters = get_filters(self.global_model)
        for client in clients:
            config = {}
            #parameters = self.personal_models[int(client.cid)]
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_evaluate_list.append((client, fit_ins))
        return config_evaluate_list
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(1, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            self.record_test_accuracy(metrics_aggregated['accuracy'])
            print(f"Fedcom: Round {server_round}, test accuracy = {metrics_aggregated['accuracy']}")
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated
    
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
  # Multiply accuracy of each client by number of examples used
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  # Aggregate and return custom metric (weighted average)
  return {"accuracy": sum(accuracies) / sum(examples)}

def fedcom_client_fn(cid) -> fedcom_client:
  Epoch = 5
  drop_rate = 0.6
  dataset = voice_dataset("clientdata/google_speech_unbalanced_client_"+str(cid)+"_ALPHA_0.1.csv")
  return fedcom_client(cid, dataset, drop_rate, Epoch, Batch)