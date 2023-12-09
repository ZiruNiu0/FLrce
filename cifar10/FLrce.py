from typing import List, Tuple, Union, Dict
from models import CNN
from FLrce_client import FLrce_client
import random
import math
import torch
from copy import deepcopy
import flwr as fl
from flwr.common import Metrics
from flwr.common import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING
from cifardataset import cifar10Dataset
from es2 import get_topk_effectiveness, max_mean_dist_split
import numpy as np
from util import get_filters, get_orthogonal_distance, compute_update, get_relationship_update_this_round, get_parameters, set_filters, highest_consensus_this_round, get_cosine_similarity, weighted_average

CHANNEL = 3
Batch = 128
DEVICE = torch.device("cpu")
MAX_EXPLOIT_RATE = 1.0
DECAY_FACTOR = 1
CLASSES = 10

class FLrce_strategy(fl.server.strategy.FedAvg):
    def __init__(self, ff, fe, mfc, mec, mac, accuracies=[], ClientsSelection=[], HighestConsensus=[], AvgConsensus=[], HCperround=[], ESCriteria=[]):
        super().__init__(fraction_fit=ff, fraction_evaluate=fe, min_fit_clients=mfc, min_evaluate_clients=mec, min_available_clients=mac, evaluate_metrics_aggregation_fn=weighted_average)
        self.fraction_fit_=ff,
        self.fraction_evaluate_=fe,
        self.min_fit_clients_=mfc,
        self.min_evaluate_clients_=mec,
        self.min_available_clients_=mac
        self.relation_map = np.zeros(mac*mac).reshape((mac, mac))
        self.consesus = np.zeros(mac*mac).reshape((mac, mac))
        self.latest_local_updates = {}
        self.latest_globalparam = {}
        self.exploremap = {}
        self.last_update_round = {}
        self.global_model = CNN(CHANNEL, outputs=CLASSES)
        self.accuracy_record = accuracies
        for i in range(mac):
            self.exploremap[str(i)] = 0
        self.selected_clients_records = ClientsSelection
        self.highest_consensus = HighestConsensus
        self.avgconsensus = AvgConsensus
        self.hcp = HCperround
        self.is_exploit_round = False
        self.stopped = False
        self.earlystopping_round = 0
        self.earlystopping_acc = 0.0
        self.early_stopping_criteria = ESCriteria
        self.EarlyStoppingCriterias = iter([4.5, 5, 5.5, 6])
        self.es_criteria = next(self.EarlyStoppingCriterias, -1)
        self.best_model = CNN(CHANNEL, outputs=CLASSES)
        self.highest_test_acc = 0.4
        self.highest_test_round = 0
        self.relation_map_saving = []
        self.earlystopping_round_2 = 999
        self.non_filter_params = {}

    """override"""
    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters(get_parameters(self.global_model))
    
    def configure_fit(
        self, server_round: int, parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        config_fit_list = []
        scoremap = self.get_effectiveness_map()
        exploremap = self.get_explore_map()
        explore_possibility = math.pow(0.98, max(server_round-1, 0))
        exploit_threshold = min(1 - explore_possibility, MAX_EXPLOIT_RATE)
        exploit_value = random.random()
        if_exploit = exploit_value <= exploit_threshold
        self.is_exploit_round = if_exploit
        clients = client_manager.sample(num_clients=sample_size, exploit_factor=if_exploit, utility_scores_map=scoremap, explore_map=exploremap, min_num_clients=min_num_clients)
        for client in clients:
            cid = int(client.cid)
            config = {}
            parameters = get_filters(self.global_model)
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_fit_list.append((client, fit_ins))
        return config_fit_list
    
    def configure_evaluate(self, server_round: int, parameters, client_manager: ClientManager):
        """override"""
        if self.fraction_evaluate_ == 0.0:
            return []
        # Parameters and config
        config = {}
        # Sample clients
        sample_size, min_num_clients = super().num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_evaluate_list = []
        for client in clients:
            parameters = get_filters(self.global_model)
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_evaluate_list.append((client, fit_ins))
        return config_evaluate_list
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        selected_clients = []
        current_parameter = get_filters(self.global_model)
        oldmap = deepcopy(self.consesus)
        updateDict = deepcopy(self.latest_local_updates)
        received_params = []
        Fitres = []
        for client, fit_res in results:
            Fitres.append(fit_res)
            cid = client.cid
            selected_clients.append(cid)
            param, num = parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples
            self.record_latest_starting_point(cid, param)
            self.set_last_update_round(cid, server_round)
            self.record_latest_local_update(cid, param, current_parameter)
            self.update_relationship(cid, param, server_round, uDict=updateDict)
            received_params.append((param, num))
        for client_id, received_parameter in [(client.cid, parameters_to_ndarrays(fit_res.parameters)) for (client, fit_res) in results]:
            self.update_current_relationship(client_id, received_parameter, results)
        self.save_relationship()
        consensus_update = get_relationship_update_this_round(results, oldmap, self.consesus)
        hc = highest_consensus_this_round(results, oldmap, self.consesus)
        self.record_selected_clients(selected_clients)
        self.record_avg_consensus(consensus_update)
        self.record_hcp(hc)
        conflicts = self.get_conflicts(results)
        if self.is_exploit_round:
            topk_effectiveness = get_topk_effectiveness(self.min_available_clients_, self.relation_map, len(results))
            if self.es_criteria > 0 and conflicts >= self.es_criteria:
                self.earlystopping_round =server_round
                self.stopped = True
                self.es_criteria = next(self.EarlyStoppingCriterias, -1)
            elif self.es_criteria > 0 and conflicts >= self.es_criteria - 0.5:
                if max_mean_dist_split(topk_effectiveness) <= 1:
                    self.earlystopping_round_2 = server_round
                    self.record_criteria_acc_round()
        parameters_aggregated = aggregate(received_params)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        set_filters(self.global_model, parameters_aggregated)
        self.highest_consensus.append(self.get_highest_consensus())
        return parameters_aggregated, metrics_aggregated
    
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
            print(f"Round {server_round}, Exploit = {self.is_exploit_round}, test accuracy = {metrics_aggregated['accuracy']}")
            if self.stopped:
                if server_round == self.earlystopping_round:
                    self.earlystopping_acc = metrics_aggregated['accuracy']
                    self.record_criteria_acc_round()
                else:
                    print(f"stopped at {self.earlystopping_round} with an test accuracy of {self.earlystopping_acc}")
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated

    def get_effectiveness_map(self):
        map = {}
        for i in range(self.min_available_clients_):
            map[str(i)] = sum(self.relation_map[i]) - self.relation_map[i][i]
        return map 
    
    def get_effectiveness_rank(self, id:str):
        rank = 1
        map = self.get_effectiveness_map()
        for i in map.keys():
            if i != id and map[i] > map[id]:
                rank += 1
        return rank

    def set_last_update_round(self, id:str, server_round):
        self.last_update_round[id] = server_round

    def get_last_update_round(self, id:str):
        return self.last_update_round[id]

    def get_explore_map(self):
        return self.exploremap
    
    def record_latest_local_update(self, id:str, local_param:List[np.ndarray], current_global_param:List[np.ndarray]):
        self.latest_local_updates[id] = compute_update(local_param, current_global_param)
    
    def record_latest_starting_point(self, id:str, starting_point:List[np.ndarray]):
        self.latest_globalparam[id] = starting_point

    def record_explore(self, cid):
        self.exploremap[cid] += 1

    def record_test_accuracy(self, acc):
        self.accuracy_record.append(acc)

    def save_relationship(self):
        new_relation_saving = []
        for i in range(self.min_available_clients_):
            relations = self.relation_map[i]
            values_to_s = [f'{r:.4f}' for r in relations]
            new_relation_saving.append(" ".join(values_to_s))
        self.relation_map_saving = new_relation_saving

    def record_criteria_acc_round(self):
        self.early_stopping_criteria.append(" ".join([str(self.es_criteria), str(self.earlystopping_round), str(self.earlystopping_acc), str(self.highest_test_round), '2e'+str(self.earlystopping_round_2)]))
    
    def record_selected_clients(self, clients:List[str]):
        self.selected_clients_records.append(" ".join(clients))

    def record_avg_consensus(self, value):
        self.avgconsensus.append(value)

    def record_hcp(self, cp):
        self.hcp.append(cp)

    def update_relationship(self, id:str, new_parameter:List[np.ndarray], server_round, uDict=None, Alpha=0.9, Decay_factor=DECAY_FACTOR):
        current_global_parameter = get_filters(self.global_model)
        new_parameter_merged = new_parameter
        this_update = compute_update(new_parameter_merged, current_global_parameter)
        if uDict == None:
            uDict = self.latest_local_updates
        for k in uDict.keys():
            if k != id:
                starting_point = self.latest_globalparam[k]
                last_round = self.get_last_update_round(k)
                new_update = compute_update(new_parameter_merged, starting_point)
                old_update = compute_update(current_global_parameter, starting_point)
                local_update = self.latest_local_updates[k]
                if server_round - last_round <= 1:
                    self.relation_map[int(id)][int(k)] = (1-Alpha)*self.relation_map[int(id)][int(k)] + Alpha * get_cosine_similarity(this_update, local_update)
                elif Decay_factor > 0.0:
                    distance1 = get_orthogonal_distance(old_update, local_update)
                    distance2 = get_orthogonal_distance(new_update, local_update)
                    new_value = max((distance1 - distance2) / (distance1 + 1e-5), -1)
                    old_value = self.relation_map[int(id)][int(k)]
                    if new_value >= old_value:
                        self.consesus[int(id)][int(k)] = 1
                    else:
                        self.consesus[int(id)][int(k)] = -1
                    self.relation_map[int(id)][int(k)] = (1-Alpha)*old_value + Alpha * new_value * math.pow(DECAY_FACTOR, server_round-last_round+1)

    def update_current_relationship(self, id:str, my_parameter, results, Alpha=0.9):
        if len(results) > 1:
            global_parameter = get_filters(self.global_model)
            this_update = compute_update(my_parameter, global_parameter)
            for client, fitres in results:
                k = client.cid
                new_local_parameter = parameters_to_ndarrays(fitres.parameters)
                local_update = compute_update(new_local_parameter, global_parameter)
                new_value = get_cosine_similarity(local_update, this_update)
                old_value = self.relation_map[int(id)][int(k)]
                if new_value >= old_value:
                    self.consesus[int(id)][int(k)] = 1
                else:
                    self.consesus[int(id)][int(k)] = -1
                self.relation_map[int(id)][int(k)] = (1-Alpha)*old_value + Alpha*new_value

    def get_highest_consensus(self) -> int:
        highest_value = -self.min_available_clients_
        for i in range(self.min_available_clients_):
            consesus_peers = sum(self.consesus[i])
            if consesus_peers >= highest_value:
                highest_value = consesus_peers
        return highest_value
    
    def get_conflicts(self, results):
        global_parameter = get_filters(self.global_model)
        total = 0
        num_clients = 0
        for client, fitres in results:
            num_clients += 1
            id = client.cid
            new_local_parameter = parameters_to_ndarrays(fitres.parameters)
            local_update = compute_update(new_local_parameter, get_filters(self.global_model))
            for k, f in results:
                if k != id:
                    local_parameter_k = parameters_to_ndarrays(f.parameters)
                    local_update_k = compute_update(local_parameter_k, global_parameter)
                    if get_cosine_similarity(local_update, local_update_k) <= 0.0:
                        total += 1
        return total / max(num_clients, 1) 

def FLrce_client_fn(cid) -> FLrce_client:
  Epoch = 5
  dataset = cifar10Dataset("clientdata/cifar10_client_"+ str(cid) + "_ALPHA_0.1.csv")
  return FLrce_client(cid, dataset, Epoch, Batch)