import Test_Utility
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator #to check if the algorithm follows the sklearn API
import pdb
import math
import os
from pathlib import Path
import platform
import matplotlib

if platform.system() == "Linux": #for matplotlib on Linux
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import utils
from datetime import timedelta
import scipy
import warnings
import itertools
from sklearn.preprocessing import StandardScaler
import time
from scipy.signal import medfilt
import multiprocessing


class EventDet_Barsim_Sequential(BaseEstimator, ClassifierMixin):
    """
           Reference implementation for the following Event Detection algorithm:
                "sequential clustering-based event detection for nonintrusive load monitoring"

                by: Karim Said Barsim and Bin Yang
                link to paper: https://pdfs.semanticscholar.org/74db/87eb3e17a2af1c4e411e2c0677ac0d20f9dc.pdf

           The algorithm implementation follows the general sklearn API.

           Use the algorithm and this class as follows:
           - create an instance of the algorithm class
           - call the fit() method
           - prepare the input window with the raw current and voltage values according to the description below
           - call the compute_input_signal() method to compute the features the algorithm requires.

           ATTENTION: It is necessary to use the compute_input_signal() function, because the raw values handed
           to the function are further used to ensure to correct input order of the input windows and to check
           if the relative offsets are in line with the ones returned from the previous window.
           The return values and offsets of the predict() function are further explained below.

           - call the predict() method on the features to detect events in the window
           - proceed with streaming the next window, depending on the result returned by the predict() method, as
           described below.

           Hence, there are three essential external methods available that are the central API of the algorithm:
           fit(), compute_input_signal() and predict().
           The fourth method _convert_relative_offset can be used to convert the offsets that are returned
           by the predict() method which are related to the input data, as computed by the compute_input_signal() function
           back to be relative to the raw input data, what is useful for the streaming process.


           Short Description of the algorithm and the input it requires:

                Input:
                    real (P) and reactive powers (Q) at a time instant, approximately computed periodewise
                    and averaged over periods.
                    We compute 2 values per second.
                    Every sample point at time t therefore has two measurements [P,Q]

                    The input that is expected by the event detector can be obtained by calling the
                    compute_input_signal method of the event detector.

                    Due to the inner workings of the algorithm, the input that is needed is longer then the length of
                    the initial input window X (with window_samples_n datapoints in it). Therefore, the input X_overall
                    that is fed to the algorithm, will be split up into two arrays, namely the initial input window X.
                    and the remaining datapoints after X, namely the X_future.
                    We recommend feeding at least as many window_samples_n datapoints that occur after the input window X to the
                    algorithm, i.e. the X_overall input should have a length of at least 2 times window_samples_n.

                    The future_window_size_n parameter that is set during initialization determines the size of the
                    X_future array.

                    As you see in the description below, the datapoints from the X_future array are appended one by one
                    two the input window X.

                    ATTENTION:
                    At the end, there are two cases, i.e. either an event is detected or not:
                    If an event is detected, two indices are returned, the beginning and the end of the event-interval
                    that was detected, else None is returned. The next fixed input window X that you should feed to the
                    algorithm, should start at the end of the event-interval (i.e. the second index + 1 that is returned).
                    So there is some overlap between the windows!
                    If no event is detected, you should continue with the next window. Most of the data contains no event,
                    hence, to speed up the algorithm, no overlap is created when no event was detected.



                The DBSCAN Algorithm is used for clustering the input data:
                    It has three Hyperparameters: Epsilon (eps) , Mininum Points (min_pts), Window Size (window_size)
                    The parameter values are not mentioned in the paper, hence we did an extensive grid search.
                    With a focus on precision, instead of recall we recommend using.
                    - eps = 0.05
                    - min_pts = 3
                    - window_size = 5


                In their paper, the authors define three event models, each of them oopses constraints a detected
                event has to fullfill. In this implementation, we use the same model the authors have used in their
                benchmark, i.e. Model 3. Model 3 is the most general one of the models, i.e. model 1 and model 2
                are specialisations with more restrictions of event-model 3.


                Event-model 3 is specified as follows:
                For the definition and the algorithm we need to define two distinct points of each cluster, that are
                important to compute multiple metrics.
                    - u: is the index of the first sample (with respect to time) n the cluster Ci (all other points that are not in the
                    cluster Ci have to have a smaller index thant u)
                    - v: is the index of the last sample (with respect to time) in the cluster Ci

                A sequence of samples X is defined as an event if:
                    (1) it contains at least two clusters C1 and C2 (besides the outlier cluster) and the outlier Cluster
                    C0 can be non empty.

                    (2) clusters C1 and C2 have a high temporal locality, i.e. Loc(Ci) >= 1 - temp_eps
                    with Loc(Ci) = n_samples_in_Ci / (v - u + 1)
                    i.e. there are at least two, non noise, clusters with a high temporal locality

                    (3) clusters C1 and C2 do not interleave in the time domain.
                    There is a point s in C1 for which all points n > s do not belong to C1 anymore, i.e. s is the upper
                    bound of C1 in this case.
                    There is also a point i in C2 for which all points n < i do not belong to C2 anymore, i.e. i is
                    the lower bound of C2 in this case.
                    (Note: changed naming here, to avoid confusion with the definition of u and v above, although
                    the meaning is the same)


                In order to fulfill these requirements we need to find at least two clusters that pass the checks.
                More then two clusters are fine. Checks (1) and (2) are performed independently, check (3)
                is performed for the remaining clusters that fullfill the pass the checks (1) and (2)

                The model constraints are implemented in the _check_event_model_constraints() method.
                If one intends to use another event model, this method has to be overwritten accordingly.

                By identifying two clusters that fulfill these requirements, we likely have discovered two
                stationary segments, consisting of sample belonging to C1 and C2. In between we have the change interval
                we are looking for, i.e. the event interval. The points in between the intervals are somehow considererd
                to be the event, if one looks closely at the samples in Figure 1 of the paper, especially in subfigure c.
                Hence all points that are in between the upper bound of C1 and the lower bound of C2, that are
                within the noise cluster (See p. 80 of the Paper). The transient is noise, that is detected by the DBSCAN.

                (Note: the upper bound of C1 < lower bound of C2, using this we can decide which cluster we name C1)
                We name points in between the event_interval_t (X_t in the paper).

                The algorithm can be divided in two steps, namely, a forward and a backward pass.
                The forward pass is used to find events, i.e. change-point intervals, and the backward pass is used
                to improve the segmentation of the input signal into steady-state and event intervals.

                1. Forward pass:
                        For a given input vector X with length n o the following:
                            1. Take the next sample x_n+1 and append it to X
                            2. Update the clustering and the clustering structure, using the DBSCAN Algorithm
                            By doing this we get clusters C1 and C2 and a possible event_interval_t
                            3. Compute the loss for the given cluster and the given event model (i.e. Model 3)
                               The loss for a signal segment X and two clusters C1 and C2 is defined as follows,
                               it counts the number of samples that need to be corrected in order for the segment
                               to match the event model:
                               L(X) = a + b + c with
                               a: number of samples n from C2 with n <= u, with u being the lower bound of
                               the event_interval_t
                               b: number of samples n from C1 with n >= v, with v being the upper bound of
                               the  event_interval_t
                               c: number of samples n between u < n < v, so number of samples n in the event_interval_t that
                               belong to C1 or C2, i.e. to the stationary signal.

                               We compute this loss for all cluster combinations, i.e. if the event model checks are
                               passed by three (non noise) clusters, then we compute two loss values

                            4. Check if L(X) <= loss_thresh.
                            If not go to step 1. and take the next sample.

                            Note: we have included a savety mechanism to prevent memory errors: if the size of X is bigger
                            then future_window_size_n times of the original window size, then we return that
                            no event was detected and the user should continue with the next input as described in the
                            input section of the documentation.

                            If yes:
                               if multiple cluster combinations have passed the loss_thresh, then declare
                               an event as detected  detected, with the change-interval event_interval_t
                               that results from the cluster combination with the lowest loss L(X)
                               Go to step 5 and start the backward pass.

                2. Backward pass:
                            1. Delete the oldest sample x1 from the segment (i.e the first sample in X)
                            2. Update the clustering and the clustering structure, using the DBSCAN Algorithm
                            3. Check the loss L(X) for the detected segment and the detected event_interval_t.
                            If L(X) <= loss_thresh, go to step 2 again.
                            If L(X=) >= loss_thresh, i.e. if without the removed sample no event is detected anymore
                            reinsert the last-sample again and declare the segment X as a balanced event

                After this is done, the process restarts from the first sample of C2 (x_v).

                The whole algorithm is window-based, with a initial window size of window_size_n
                The event detector has the following hyperparameters that can be fine-tuned.
                    - DBSCAN Epsilon (dbscan_eps)
                    - DBSCAN Mininum Points (dbscan_min_pts)
                    - Window Size (window_size_n)
                    - Threshold for the Loss-Function (loss_thresh)
                    - Temporal Locality Epsilon (temp_eps)

                Non algorithm related parameters that can be adapted are:
                    - Number of datapoints after the input window that are fed to the algorithm future_window_size_n
                    - Flag to turn input window order checks on or of perform_input_order_checks
                    (see also the details in the input section of this documentation for the two parameters above)

                If you want to debug the inner workings of the algorithm, i.e. get plots on the clustering etc. .
                then initialize the estimator with debugging_mode=False.
                Using this in a graphical environment, like a jupyter notebook is highly recommended.
                This provides a tool to understand the inner workings of the algorithm in detail.
           """

    def __init__(self, dbscan_eps=0.05, dbscan_min_pts=3, window_size_n=5, future_window_size_n=5,
                 loss_thresh=40, temp_eps=0.8, debugging_mode=False, dbscan_multiprocessing=False, network_frequency=60, **kwargs):
        """


        Args:
            dbscan_eps (float): Epsilon Parameter for the DBSCAN algorithm
            dbscan_min_pts (int): Minimum Points Parameter for the DBSCAN algorithm
            window_size_n (int): Window Size
            future_window_size_n (int): Maximum Number of samples that are gradually added to the window_size_n window.
            window_size_n +n future_window_size_n is the maximum possible window, then no event detected is returned
            in case.
            loss_thresh (int): trehshold fopr the loss-function
            temp_eps (float):  temporal locality epsilon
            perform_input_order_checks: check the correct input order before processing the input, as described in
            the doc-string
            debugging_mode (bool): activate if plots of the dbscan clustering shall be shown
            grid_search_mode: activate to adapt the score function, if you want to perfrom grid-search
            dbscan_multiprocessing (bool): default=False, if set to true multiple processes are used in the dbscan algorithm.
            If the Barsim_Sequential event detector is used within a multiprocessing environment, turning the dbscan_multiprocessing
            paramter to True, results in warnings by sklearn and the multiprocessing library, as no additional subprocesses can
            be spawned by the processes.


        """


        self.dbscan_eps = dbscan_eps
        self.dbscan_min_pts = dbscan_min_pts
        self.window_size_n = window_size_n
        self.future_window_size_n = future_window_size_n
        self.loss_thresh = loss_thresh
        self.temp_eps = temp_eps

        self.network_frequency = network_frequency # periods per second

        # initialize the corresponding parameter for the checks
        self.order_safety_check = None



        self.debugging_mode = debugging_mode

        self.dbscan_multiprocessing=dbscan_multiprocessing

        # Display some warning if the window_size_n and the future_window_size_n is unbalanced
        if future_window_size_n / window_size_n < 5:
            warnings.warn("You have provided less then 5 times of future_window_samples_n than window_size_n samples "
                          "to the algorithm. Please make sure that your window_size_n is at least big enough for the "
                          "event detector to work. We recommend using more future samples", Warning)

    def fit(self):
        """
        Call before calling the predict function. Needed by the sklearn API.

        """

        self.is_fitted = True

    def predict(self, X_overall):
        """
        Predict if the input provided contains an event or not.
        The input provided should be computed by the compute_input_signal function of this class.
        Args:
            X_overall (ndarray): Input computed by compute_input_signal function.

        Returns:
            event_interval_indices (tuple): (start_index, end_index), if no event detected None is returned

        """
        # Check if fit was called before
        check_is_fitted(self, ["is_fitted"])

        # 1. Check the input
        # 1.1 Perform general tests on the input array

        utils.assert_all_finite(X_overall)
        X_overall = utils.as_float_array(X_overall)

        # 1.2 Check the size of X_overall
        if not len(X_overall) == self.window_size_n + self.future_window_size_n:
            raise ValueError("The input size of X_overall (" + str(len(X_overall)) + ") does not match the"
             "window_size_n (" + str(self.window_size_n) + ") + the future_window_samples_n (" +
                               str(self.future_window_size_n) + ") parameters")


        # 1.3 Split the input array into the initial inout window X and the remaining X_future datapoints
        X, X_future = np.split(X_overall, [self.window_size_n])


        # 2. Event Detection Logic

        # 2.1 Forward Pass

        # We need to loop over all samples in X_future, until an event is detected
        # In case no event is detected and all samples in X_future are added to X, we return None

        # 2.1.1 Take the next sample x_n+1 and append it to X
        if self.debugging_mode == True:
            processing_start_time = time.process_time()

            self.datapoints_added = 0

        for new_datapoint in X_future:
            if self.debugging_mode == True:
                self.datapoints_added += 1

            event_detected = False # Flag to indicate if an event was detected or not

            X = np.concatenate([X, [new_datapoint]])

            # 2.1.2 Update the clustering and the clustering structure, using the DBSCAN Algorithm
            # By doing this we get clusters C1 and C2
            self._update_clustering(X)

            # Now check the mode constraints
            # Possible intervals event_interval_t are computed in the _check_event_model_constraints() function.
            checked_clusters = self._check_event_model_constraints()

            # If there are no clusters that pass the model constraint tests, the method _check_event_model_constraints()
            # returns None, else a list of triples (c1, c2, event_interval_t).

            if checked_clusters is None:
                continue #add the next new_datapoint. We go back to step 1 of the forward pass.

            # 2.1.3 Compute the Loss-values

            else: # at least one possible combination of two clusters fullfills the event model constraints-
                # Hence, we can proceed with step 3 of the forward pass.
                # Therefore, we compute the loss for the given cluster combination.
                # The formula for the loss, is explained in the doc-string in step 3 of the forward pass.

                event_cluster_combination = self._compute_and_evaluate_loss(checked_clusters)
                self.forward_clustering_structure = self.clustering_structure #save the forward clustering structure

                if event_cluster_combination is not None: #  event detected
                    event_detected = True
                    break #leave the loop of adding new samples each round and continue the code after the loop

                else: # go back to step 1 and add the next sample
                    continue


        if event_detected == True: #an event was detected in the forward pass, so the backward pass is started
            if self.debugging_mode == True:
                print("Event Detected at: " + str(event_cluster_combination))
                print("")
                print("")
                print(60*"*")
                print("Backward pass is starting")
                print(60 * "*")

            # Initialize the backward pass clustering with the forward pass clustering, in case already the
            # first sample that is removed, causes the algorithm to fail. Then the result from the forward
            # pass is the most balanced event
            self.backward_clustering_structure = self.forward_clustering_structure
            event_cluster_combination_balanced = event_cluster_combination

            # 2.2.1. Delete the oldest sample x1 from the segment (i.e the first sample in X)
            for i in range(1, len(X)-1):
                X_cut = X[i:] #delete the first i elements, i.e. in each round the oldest sample is removed

                # 2.2.2 Update the clustering structure
                self._update_clustering(X_cut) #the clustering_structure is overwritten, but the winning one
                # from the forward pass is still saved in the forward_clustering_structure attribute

                # 2.2.3 Compute the loss-for all clusters that are detected (except the detected)
                # Hence, we need to check the event model constraints again
                checked_clusters = self._check_event_model_constraints()

                if checked_clusters is None: #roleback with break
                    status = "break"
                    event_cluster_combination_balanced = self._roleback_backward_pass(status, event_cluster_combination_balanced,i)
                    break #finished

                else: #compute the loss
                    # 2.2.4 Check the loss-values for the detected segment
                    event_cluster_combination_below_loss = self._compute_and_evaluate_loss(checked_clusters)

                    if event_cluster_combination_below_loss is None: #roleback with break
                        status = "break"
                        event_cluster_combination_balanced = self._roleback_backward_pass(status,
                                                                                          event_cluster_combination_balanced,
                                                                                          i)
                        break #finished
                    else:
                        status = "continue"
                        event_cluster_combination_balanced = self._roleback_backward_pass(status,
                                                                                          event_cluster_combination_balanced,
                                                                                          i,
                                                                                          event_cluster_combination_below_loss
                                                                                          )
                        continue #not finished, next round, fiight

            event_start = event_cluster_combination_balanced[2][0]
            event_end = event_cluster_combination_balanced[2][-1]
            if self.debugging_mode == True:
                print("Balanced event detected in the Backward pass from " + str(event_start) + " to " + str(event_end))
            # In case an event is detected, the first sample of the second steady state segment (c2) should be fed to
            # the estimator again for further event detection, as described in the documentation.
            # We use the first 10 start values to perform the necessary input check if the corresponding parameter
            # perform_input_order_checks = True


            self.order_safety_check = {"first_10_start_values" : X[event_end + 1: event_end + 11] }

            if self.debugging_mode is True:
                elapsed_time = time.process_time() - processing_start_time
                print("")
                print("Processing this window took: " + str(elapsed_time) + " seconds")



            return (event_start, event_end)
        else:
            # also for the input order check
            # in case no event is detected, the user should feed back the last window_size_n samples of X.
            # this is implemented that way to prevent memory issues
            self.order_safety_check = {"first_10_start_values": X[-self.window_size_n:][:10]}

            if self.debugging_mode is True:
                elapsed_time = time.process_time() - processing_start_time
                print("")
                print("Processing this window took: " + str(elapsed_time) + " seconds")

            return None

    def _compute_and_evaluate_loss(self, checked_clusters):
        """
        Function to compute the loss values of the different cluster combinations.
        The formula for the loss, is explained in the doc-string in step 3 of the forward pass.

        Args:
            checked_clusters (list): of triples (c1, c2, event_interval_t)

        Returns:
            event_cluster_combination (tuple): triple of the winning cluster combination

        """

        if self.debugging_mode is True:
            print("")
            print("")
            print("Compute the Loss values for all cluster combinations that have passed the model constraints")
            print("They have to be smaller than: " + str(self.loss_thresh))

        event_model_loss_list = []
        for c1, c2, event_interval_t in checked_clusters:
            lower_event_bound_u = event_interval_t[0] - 1  # the interval starts at u + 1
            upper_event_bound_v = event_interval_t[-1] + 1  # the interval ends at v -1
            c1_indices = self.clustering_structure[c1]["Member_Indices"]
            c2_indices = self.clustering_structure[c2]["Member_Indices"]
            a = len(np.where(c2_indices <= lower_event_bound_u)[0]) # number of samples from c2 smaller than lower bound of event

            b = len(np.where(c1_indices >= upper_event_bound_v)[0]) # number of samples from c1 greater than upper bound of event

            c1_and_c2_indices = np.concatenate([c1_indices, c2_indices])

            # number of samples n between u < n < v, so number of samples n in the event_interval_t that
            # belong to C1 or C2, i.e. to the stationary signal.
            c = len(np.where((c1_and_c2_indices > lower_event_bound_u) & (c1_and_c2_indices < upper_event_bound_v))[0])


            event_model_loss = a + b + c

            event_model_loss_list.append(event_model_loss)

            if self.debugging_mode is True:
                print("\tLoss for clusters " + str(c1) + " and " + str(c2) + ": " + str(event_model_loss))
                print("\t\tComposited of: " + "a=" + str(a) + " b=" + str(b) + " c=" +str(c))

        # 2.1.4 Compare loss value to the threshold on the loss loss_thresh
        # We select the cluster combination with the smallest loss, that is below the threshold

        # Select the smallest loss value
        min_loss_idx = np.argmin(event_model_loss_list)  # delivers the index of the element with the smallest loss

        # Compare with the loss threshold, i.e. if the smallest loss is not smaller than the treshold, no other
        # loss will be in the array

        if event_model_loss_list[min_loss_idx] <= self.loss_thresh:  # if smaller than the threshold event detected
            event_cluster_combination = checked_clusters[min_loss_idx]  # get the winning event cluster combination
            if self.debugging_mode is True:
                print("\tEvent Cluster Combination determined")
            return event_cluster_combination

        else:
            return None

    def _update_clustering(self, X):
        """
        Using the DBSCAN Algorithm to update the clustering structure.
        All available CPUs are used to do so.
        Furthermore all relevant metrics are directly computed from the clustering result.

        The method sets the clustering_structure attribute of the estimator class:
            clustering_structure (dict): resulting nested clustering structure. contains the following keys
            For each cluster it contains: {"Cluster_Number" : {"Member_Indices": []"u" : int,"v" : int,"Loc" : float} }
            u and v are the smallest and biggest index of each cluster_i respectively.
            Loc is the temporal locality metric of each cluster_i.


        Args:
            X (ndarray): input window, shape=(n_samples, 2)

        Returns:
            None


        """



        # Do the clustering
        # Use all CPU's for this, i.e. set n_jobs = -1
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts, n_jobs=-1).fit(X)

        # Get the cluster labels for each datapoint in X
        X_cluster_labels = np.array(dbscan.labels_)

        # Noise samples get the "-1" class --> those are usually the transients
        # Get all unique cluster identifiers
        cluster_labels = np.unique(X_cluster_labels)


        if self.debugging_mode == True: #if in debuggin mode, plot the clusters

            if self.original_non_log is False:
                log_label = "(Log-Scale)"
            else:
                log_label = ""

            plt.clf()
            plt.scatter(x=np.arange(len(X)),y=X[:, 0], c=X_cluster_labels, cmap='Paired')
            plt.ylabel("Active Power " + log_label)
            plt.xlabel("Samples")
            plt.title("Clustering")
            plt.show()
            plt.clf()
            plt.scatter(x=X[:, 1], y=X[:,0],  c=X_cluster_labels, cmap='Paired')
            plt.ylabel("Active Power " + log_label)
            plt.xlabel("Reactive Power " + log_label)
            plt.title("Clustering")
            plt.show()

        clustering_structure = {}

        #build the cluster structure, for each cluster store the indices of the points.
        for cluster_i in cluster_labels:
            cluster_i_structure = {} #all the relvant information about cluster_i

            # Which datapoints (indices) belong to cluster_i
            cluster_i_member_indices = np.where(X_cluster_labels == cluster_i)[0]
            cluster_i_structure["Member_Indices"] = np.array(cluster_i_member_indices)

            # Determine u and v of the cluster (the timely first and last element, i.e. the min and max index)
            u = np.min(cluster_i_member_indices)
            v = np.max(cluster_i_member_indices)
            cluster_i_structure["u"] = u
            cluster_i_structure["v"] = v

            # compute the temporal locality of cluster_ci
            Loc_cluster_i = len(cluster_i_member_indices) / (v - u + 1) # len(cluster_i_member_indices) = n_samples_in_Ci
            cluster_i_structure["Loc"] = Loc_cluster_i

            # insert the structure of cluster_i into the overall clustering_structure
            clustering_structure[cluster_i] = cluster_i_structure


        self.clustering_structure = clustering_structure

        return None

    def _roleback_backward_pass(self, status, event_cluster_combination_balanced, i, event_cluster_combination_below_loss=None):
        """
        When the backward pass is performed, the oldest datapoint is removed in each iteration.
        After that, first the model constraints are evaluated.
        If they are violated, we roleback to the previous version by adding the oldest datapoint again
        and we are finished.
        In case the model constraints still hold, we recompute the loss.
        If the loss exceeds the threshold, we ne to roleback to the last version too.

        This roleback happens at to positions in the code (i.e. after the model constraints are evaluated and after
        the loss computation). Therefore, it is encapsulated in this function.

        Args:
            status (string): either "continue" or "break"
            i: current iteration index of the datapoint
            event_cluster_combination_balanced:
            event_cluster_combination_below_loss:

        Returns:

        """
        if status == "break":
            # if the loss is above the threshold
            # without the recently removed sample, take the previous combination and declare it as an
            # balanced event.
            # the previous clustering and the previous event_cluster_combination are saved from the previous
            # run automatically, so there is no need to perform the clustering again.

            # Attention: the event_interval indices are now matched to X_cut.
            # We want them to match the original input X instead.
            # Therefore we need to add + (i-1) to the indices, the  -1 is done because we take
            # the clustering and the state of X_cut from the previous, i.e. i-1, round.
            # This is the last round where the loss, was below the threshold, so it is still fine
            event_cluster_combination_balanced = list(event_cluster_combination_balanced)
            event_cluster_combination_balanced[2] = event_cluster_combination_balanced[2] + i  # the event_interval_t
            event_cluster_combination_balanced = tuple(event_cluster_combination_balanced)


            # The same is to be done for all the final cluster
            # The structure stored in self.backward_clustering_structure is valid, it is from the previous iteration
            for cluster_i, cluster_i_structure in self.backward_clustering_structure.items():
                cluster_i_structure["Member_Indices"] = cluster_i_structure["Member_Indices"] + int(i - 1)
                cluster_i_structure["u"] = cluster_i_structure["u"] + int(i - 1)
                cluster_i_structure["v"] = cluster_i_structure["v"] + int(i - 1)

                # Only the "Loc" is not updated (stays the same, regardless of the indexing)
                self.backward_clustering_structure[cluster_i] = cluster_i_structure

            return event_cluster_combination_balanced

        elif status == "continue":  # continue with the backward pass
            # update the backward_clustering_structure with the latest valid one
            # i.e. the new clustering structure

            self.backward_clustering_structure = self.clustering_structure
            event_cluster_combination_balanced = event_cluster_combination_below_loss  # same here
            return event_cluster_combination_balanced

        else:
            raise ValueError("Status code does not exist")

    def _check_event_model_constraints(self):
        """
        Checks the constraints the event model, i.e. event model 3, opposes on the input data.
        It uses the clustering_structure attribute, that is set in the _update_clustering() function.

        Arguments:

        Returns:
            checked_clusters (list): list of triples (c1, c2, event_interval_t)
                                    with c1 being the identifier of the first cluster, c2 the second cluster
                                    in the c1 - c2 cluster-combination, that have passed the model
                                    checks. The event_interval_t are the indices of the datapoints in between the two
                                    clusters.


        """

        if self.debugging_mode is True:
            print("")
            print("Perform check 1 to find non noise cluster")

        # (1) it contains at least two clusters C1 and C2, besides the outlier cluster, and the outlier Cluster C0
        # can be non empty. (The noisy samples are given the the cluster -1 in this implementation of DBSCAN)
        n_with_noise_clusters = len(self.clustering_structure)
        n_non_noise_clusters = n_with_noise_clusters - 1 if -1 in self.clustering_structure.keys() else n_with_noise_clusters

        if self.debugging_mode is True:
            print("Number of non noise_clusters: " + str(n_non_noise_clusters))

        if n_non_noise_clusters < 2: #check (1) not passed
            return None

        # If check (1) is passed, continue with check (2)

        # (2) clusters C1 and C2 have a high temporal locality, i.e. Loc(Ci) >= 1 - temp_eps
        # i.e. there are at least two, non noise, clusters with a high temporal locality

        check_two_clustering = {} #store the clusters that pass the test in a new structure

        if self.debugging_mode is True:
            print("")
            print("Perform check 2 with temp locality greater than " + str(1 - self.temp_eps))
            print("Cluster | Temporal Locality")
            print("--------|----------------- ")
        for cluster_i, cluster_i_structure in self.clustering_structure.items():
            if cluster_i != -1: # for all non noise clusters

                if self.debugging_mode is True:
                   print(str(cluster_i) + "       | " + str(cluster_i_structure["Loc"]))

                if cluster_i_structure["Loc"] >= 1 - self.temp_eps: #the central condition of condition (2)
                    check_two_clustering[cluster_i] = cluster_i_structure


        if self.debugging_mode is True:
            print("Number of clusters that pass temporal locality epsilon(Check 2): " + str(n_non_noise_clusters) + " (min 2 clusters) ")

        if len(check_two_clustering) < 2:  #check (2) not passed
            return None

        # (3) two clusters C1 and C2 do not interleave in the time domain.
        # There is a point s in C1 for which all points n > s do not belong to C1 anymore.
        # There is also a point i in C2 for which all points n < i do not belong to C2 anymore.
        # i.e. the maximum index s of C1 has to be smaller then the minimum index of C2
        checked_clusters = []

        # We need to compare all pairs of clusters in order to find all pairs that fullfill condition (3)

        if self.debugging_mode is True:
            print("")
            print("Perform check 3 for all combinations ")

        cluster_combinations = itertools.combinations(check_two_clustering, 2) #get all combinations, without replacement

        for cluster_i, cluster_j in cluster_combinations: #for each combinations

            # the cluster with the smaller u, occurs first in time, we name it C1 according to the paper terminology here
            if check_two_clustering[cluster_i]["u"] < check_two_clustering[cluster_j]["u"]:
                #  #cluster_i is starting before cluster_j
                c1 = cluster_i
                c2 = cluster_j
            else: #then cluster_j is starting before cluster i
                c1 = cluster_j
                c2 = cluster_i

            # now we check if they are overlapping
            # the maximum index of C1 has to be smaller then the minimum index of C2, then no point of C2 is in C1
            # and the other way round, i.e all points in C1 have to be smaller then u of C2

            if check_two_clustering[c1]["v"] < check_two_clustering[c2]["u"]: #no overlap detected
                # if thee clusters pass this check, we can compute the possible event_interval_t (i.e. X_t)
                # for them. This interval possibly contains an event and is made up of all points between cluster c1
                # and cluster c2 that are noisy-datapoints, i.e. that are within the -1 cluster. THe noisy points
                # have to lie between the upper-bound of c1 and the lower-bound of c2
                if -1 not in self.clustering_structure.keys():
                    return None
                else:
                    c0_indices = self.clustering_structure[-1]["Member_Indices"] #indices in the -1 (i.e. noise) cluster


                #ASSUMPTION if there is no noise cluster, then the check is not passed.
                # No event segment is between the two steady state clusters then.
                # Any other proceeding would cause problems in all subsequent steps too.

                if self.debugging_mode is True:
                    print("No overlap between cluster " + str(c1) + " and " + str(c2) + " (i.e. a good Candidate)")
                    print("\tPotential event window (noise cluster indices:  " + str(c0_indices))
                    print("\tCluster 1 v: " + str(check_two_clustering[c1]["v"] ))
                    print("\tCluster 2 u: " + str(check_two_clustering[c2]["u"]))

                # check the condition, no overlap between the noise and steady state clusters allowed: u is lower, v upper bound
                condition = [(c0_indices > check_two_clustering[c1]["v"]) & (c0_indices < check_two_clustering[c2]["u"])]
                event_interval_t = c0_indices[condition]

                if self.debugging_mode is True:
                    print("\tEvent Intervall between cluster " + str(c1) + " and " + str(c2) + " with indices " + str(event_interval_t))

                # If the event_interval_t contains no points, we do not add it to the list too,
                # i.e. this combinations does not contain a distinct event interval between the two steady state
                # cluster sections.
                if len(event_interval_t) != 0:

                    checked_clusters.append((c1, c2, event_interval_t))

                else:
                    if self.debugging_mode is True:
                        print("Check 3: Event interval between the steady states was empty")
        # now all non interleaving clusters are in the check_three_clustering structure
        # If there are still at least two non-noise clusters in check_three_clustering, then check (3) is passed,
        # else not.


        if self.debugging_mode is True:
            print("Number of cluster-pairs that pass Check 3: " + str(len(checked_clusters)) + " (min 1 pair)")

        if len(checked_clusters) < 1:  # check (3) not passed


            return None


        return checked_clusters

    def compute_input_signal(self, voltage, current, period_length, original_non_log=False):
        """
        This functions uses the instantaneous voltage and current signals to compute the real (P) and reactive power (Q).
        The period_length has to divide length of the input signals evenly, i.e. no remainder.
        If this is not the case, an exception is raised.
        This is because the features are computed on a period-based approximation.

        The features are computed per 0.5 seconds, thus, two features per second are computed.

        ASSUMPTION: To reduce the variation in the input for the clustering, we logarthmize the data as the
        authors have done in other publications by them. In this paper, they do not detail the pre-processing and
        input releated hyperparameters like the window size.
        If you do not agree with this assumption, set the original_non_log parameter to True, and the signal
        will not be converted to log-scale.

        Args:
            voltage (ndarray): one-dimensional voltage array
            current (ndarray: one-dimensional current array
            period_length (int): length of a period for the given dataset.
                           example: sampling_rate=10kHz, 50Hz basefrequency,
                           10kHz / 50 = 200 samples / period
            original_non_log (bool): default: False, if set to true, the non-logarithmized data is returned.


        Returns:
            X (ndarray): feature vector with active and reactive power, with shape(window_size_n,2).
                         The first component at time t is the active, the second one the reactive power.
                         The active and reactive power are per values per second.



        """

        voltage, current = utils.check_X_y(voltage, current, force_all_finite=True, ensure_2d=False,allow_nd=False, y_numeric=True,
                  estimator="EventDet_Barsim_Sequential")

        period_length = int(period_length)

        compute_features_window = period_length*self.network_frequency/2 #compute features averaged over this timeframe

        # also ensure that the input length corresponds to full seconds
        if len(voltage) % (int(period_length) * self.network_frequency) != 0:
            raise ValueError("Ensure that the input signal can be equally divided with the period_length and full seconds! "
                             "The input has to correspond to full seconds, e.g. lengths such as 1.3 seconds are not permitted")
        self.original_non_log = original_non_log

        #Compute the active and reactive power using the Metrics class provided with this estimator
        Metrics = Test_Utility.Electrical_Metrics()

        active_power_P= Metrics.active_power(instant_voltage=voltage, instant_current=current,
                                            period_length=compute_features_window) #values per second

        apparent_power_S = Metrics.apparent_power(instant_voltage=voltage, instant_current=current,
                                            period_length=compute_features_window) #values per second


        reactive_power_Q = Metrics.reactive_power(apparent_power=apparent_power_S, active_power=active_power_P)

        self.period_length = compute_features_window #used to convert the offsets back to the original data

        #Now we combine the two features into a signel vector of shape (window_size_n,2)
        X = np.stack([active_power_P, reactive_power_Q], axis=1)

        if original_non_log == False:

            X = np.log(X) #logarithmize the signal as described in the assumption

        return X

    def _convert_relative_offset(self, relative_offset, raw_period_length=None):
        """
        Convert the relative offset that is computed relative to the input of the algorithm, i.e. the return
        value of the compute_input_signal() function.

        This utility function can be used to adapt the offset back to the raw data.

        "To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method "


        Args:
            relative_offset (int): the offset that needs to be converted
            raw_period_length (int): length in samples of one period in the original (the target) raw data

        Returns:
            target_offset (int): offset relative to the raw (target) input
        """

        if raw_period_length is None:
            raw_period_length = self.period_length

        if raw_period_length is None:
            raise ValueError("To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method ")

        target_offset = relative_offset * raw_period_length

        return target_offset

    def _convert_index_to_timestamp(self, index, start_timestamp_of_window):
        """
        Function to convert an index that is relative to the start_timestamp of a window that was computed
        by the compute_input_signal function to a timestamp object.

        Args:
            index (int): index to convert, relative to the input window. Features that have been used to do the event
            detection and to get the index, have to be computed according to the compute_input_signal function.
            Otherwise the timestamps returned by this function can be wrong.

            start_timestamp_of_window(datetime): start timestamp of the window the event index is located in.
            network_frequency(int): basic network frequency
        Returns:

            event_timestamp (datetime)

        """

        seconds_since_start = index / 2# 2 values per second
        event_timestamp = start_timestamp_of_window + timedelta(seconds=seconds_since_start)

        return event_timestamp

    @staticmethod
    def score(ground_truth_events_p, detected_events_p, number_of_samples_in_dataset_p=None,
              tolerance_limit_sec_p=1, return_event_lists_p=False):
        """
        This function can be used to determined the values of the confusion matrix.

        It works as follows:
        For each of the detected events, we build an interval around the detected timestamp using the tolerance_limit_sec
            Then we check for each of the detected intervals if we find a ground truth event that falls within this intervall
             If this is the case we have found a True Positive Event (TP) :
               1. TP += 1
               2. Eliminate this event from the ground truth list, in order to: avoid double detection
               In the case there is more then one ground truth events that fits into the interval, we take the first one
             If this is not the case we have found a False Positive Event (FP):
               1. FP += 1

             After this procedure: all events that are left in the ground truth list have not been detected by the event detection algorithm
             (because they were not removed from the in the previous step)
             Hence those events are actually true, but are delcared to be false by the algorithm: They are False Negatives (FN)
             For each of those events left in the ground truth list: FN += 1

             Finally we determine the number of True Negative Events(TN):
             Lastly, when all the detected and groundtruth events have been processed, the TN are calculated by
             subtracting the TP, FN, and FP from the number of samples in the dataset, i.e., all the positions where an event could have happened
             The start and the end time of the dataset and the information about sampling rate is used to compute the total
             number of possible event timestamps.

        Parameters
        ----------
        ground_truth_events_p: list
            List of datetime objects, containing the ground truth event timestamps.

        detected_events_p: list
            List of datetime objects, containing the event detected by the event detector

        number_of_samples_in_dataset_p: int
            Number of samples, i.e. number of possible event positions in the dataset.
            When computing period wise features, like RMS, values use the corresponding number of samples.
            The value is used to compute the number of True Negative Events.

        tolerance_limit_sec_p: int, optional (default = 1)
            The tolerance limit that is used to determine the scores of the confusion matrix.

        return_events_list_p: bool, optional (default = False)
            If set to True, the events that are TP, FP, and FN are returned in the results dictionary
        Returns
        -------

        if grid_search_mode is set to True, only the f1score is returned, otherwise:

        results: dictionary
            A dictionary containing the following keys:  tp, fp, fn, tn, f1score, recall, precision, fpp, fpr
            With being:
            tp = True Positives
            fp = False Positives
            fn = False Negatives
            tn = True Negatives
            f1score = F1-Score Metric
            recall = Recall Metric
            precision = Precision Metric
            fpp = False Positive Percentage Metric
            fpr = False Positive Rate Metric

            If return_events_list_p is set to True, the results also contain the following keys:
            tn_events = events that are tn
            fp_events = events that are fp
            fn_events = events that are fn

        """

        ground_truth_events = ground_truth_events_p #ground_truth_events_p.copy() #copy the events
        detected_events = detected_events_p.copy() #copy the events

        results = {}
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        TP_events = []
        FP_events = []
        FN_events = []


        # Compute the Tolerance Intervals around the detected events and iterate through the detected events
        for detected_event in detected_events:
            upper_limit = detected_event + timedelta(seconds=tolerance_limit_sec_p)
            lower_limit = detected_event - timedelta(seconds=tolerance_limit_sec_p)

            # Check which ground_truth events lie within this interval
            matching_ground_truth_events = [(index, ground_truth_event) for index, ground_truth_event in
                                            enumerate(ground_truth_events) if
                                            lower_limit <= ground_truth_event <= upper_limit]

            if len(matching_ground_truth_events) > 0:  # at least (minimum) one ground truth event was detected
                TP += 1
                first_index = matching_ground_truth_events[0][0]

                TP_events.append(ground_truth_events[first_index])
                # It can be possible that there are multiple ground truth events within this interval, hence,
                # we only consider the first one and remove it from the list
                del ground_truth_events[first_index]

            else:  # No Matching Event was found
                FP_events.append(detected_event)
                FP += 1

        # Use the remaining ground truth events to determine the number of FN
        FN = len(ground_truth_events)
        FN_events = ground_truth_events.copy()

        # Use the number of samples in the dataset to determine the TNs in the dataset
        # For BLUED we have data from 20/10/2011 11:58:32.623499 to 27/10/2011 14:10:40.0

        TN = number_of_samples_in_dataset_p - (TP + FP + FN)
        # Compute the final scores
        fpp = FP / (TP + FN)
        fpr = FP / (FP + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * precision * recall) / (precision + recall)

        results["tp"] = TP
        results["fp"] = FP
        results["fn"] = FN
        results["tn"] = TN
        results["f1score"] = f1score
        results["recall"] = recall
        results["precision"] = precision
        results["fpp"] = fpp
        results["fpr"] = fpr

        if return_event_lists_p is True:
            results["tp_events"] = TP_events
            results["fp_events"] = FP_events
            results["fn_events"] = FN_events

        return results

class STREAMING_EventDet_Barsim_Sequential(BaseEstimator, ClassifierMixin):
    """
           Reference implementation for the following Event Detection algorithm:
                "sequential clustering-based event detection for nonintrusive load monitoring"

                by: Karim Said Barsim and Bin Yang
                link to paper: https://pdfs.semanticscholar.org/74db/87eb3e17a2af1c4e411e2c0677ac0d20f9dc.pdf

           The algorithm implementation follows the general sklearn API.

           Use the algorithm and this class as follows:
           - create an instance of the algorithm class
           - call the fit() method
           - prepare the input window with the raw current and voltage values according to the description below
           - call the compute_input_signal() method to compute the features the algorithm requires.

           ATTENTION: It is necessary to use the compute_input_signal() function, because the raw values handed
           to the function are further used to ensure to correct input order of the input windows and to check
           if the relative offsets are in line with the ones returned from the previous window.
           The return values and offsets of the predict() function are further explained below.

           - call the predict() method on the features to detect events in the window
           - proceed with streaming the next window, depending on the result returned by the predict() method, as
           described below.

           Hence, there are three essential external methods available that are the central API of the algorithm:
           fit(), compute_input_signal() and predict().
           The fourth method _convert_relative_offset can be used to convert the offsets that are returned
           by the predict() method which are related to the input data, as computed by the compute_input_signal() function
           back to be relative to the raw input data, what is useful for the streaming process.


           Short Description of the algorithm and the input it requires:

                Input:
                    real (P) and reactive powers (Q) at a time instant, approximately computed periodewise
                    and averaged over periods.
                    We compute 2 values per second.
                    Every sample point at time t therefore has two measurements [P,Q]

                    The input that is expected by the event detector can be obtained by calling the
                    compute_input_signal method of the event detector.



                    ATTENTION:
                    At the end, there are two cases, i.e. either an event is detected or not:
                    If an event is detected, two indices are returned, the beginning and the end of the event-interval
                    that was detected, else None is returned. The next fixed input window X that you should feed to the
                    algorithm, should start at the end of the event-interval (i.e. the second index + 1 that is returned).
                    So there is some overlap between the windows!
                    If no event is detected, you should continue with the next window. Most of the data contains no event,
                    hence, to speed up the algorithm, no overlap is created when no event was detected.



                The DBSCAN Algorithm is used for clustering the input data:
                    It has three Hyperparameters: Epsilon (eps) , Mininum Points (min_pts), Window Size (window_size)
                    The parameter values are not mentioned in the paper, hence we did an extensive grid search.
                    With a focus on precision, instead of recall we recommend using.
                    - eps = 0.05
                    - min_pts = 3
                    - window_size = 5


                In their paper, the authors define three event models, each of them oopses constraints a detected
                event has to fullfill. In this implementation, we use the same model the authors have used in their
                benchmark, i.e. Model 3. Model 3 is the most general one of the models, i.e. model 1 and model 2
                are specialisations with more restrictions of event-model 3.


                Event-model 3 is specified as follows:
                For the definition and the algorithm we need to define two distinct points of each cluster, that are
                important to compute multiple metrics.
                    - u: is the index of the first sample (with respect to time) n the cluster Ci (all other points that are not in the
                    cluster Ci have to have a smaller index thant u)
                    - v: is the index of the last sample (with respect to time) in the cluster Ci

                A sequence of samples X is defined as an event if:
                    (1) it contains at least two clusters C1 and C2 (besides the outlier cluster) and the outlier Cluster
                    C0 can be non empty.

                    (2) clusters C1 and C2 have a high temporal locality, i.e. Loc(Ci) >= 1 - temp_eps
                    with Loc(Ci) = n_samples_in_Ci / (v - u + 1)
                    i.e. there are at least two, non noise, clusters with a high temporal locality

                    (3) clusters C1 and C2 do not interleave in the time domain.
                    There is a point s in C1 for which all points n > s do not belong to C1 anymore, i.e. s is the upper
                    bound of C1 in this case.
                    There is also a point i in C2 for which all points n < i do not belong to C2 anymore, i.e. i is
                    the lower bound of C2 in this case.
                    (Note: changed naming here, to avoid confusion with the definition of u and v above, although
                    the meaning is the same)


                In order to fulfill these requirements we need to find at least two clusters that pass the checks.
                More then two clusters are fine. Checks (1) and (2) are performed independently, check (3)
                is performed for the remaining clusters that fullfill the pass the checks (1) and (2)

                The model constraints are implemented in the _check_event_model_constraints() method.
                If one intends to use another event model, this method has to be overwritten accordingly.

                By identifying two clusters that fulfill these requirements, we likely have discovered two
                stationary segments, consisting of sample belonging to C1 and C2. In between we have the change interval
                we are looking for, i.e. the event interval. The points in between the intervals are somehow considererd
                to be the event, if one looks closely at the samples in Figure 1 of the paper, especially in subfigure c.
                Hence all points that are in between the upper bound of C1 and the lower bound of C2, that are
                within the noise cluster (See p. 80 of the Paper). The transient is noise, that is detected by the DBSCAN.

                (Note: the upper bound of C1 < lower bound of C2, using this we can decide which cluster we name C1)
                We name points in between the event_interval_t (X_t in the paper).

                The algorithm can be divided in two steps, namely, a forward and a backward pass.
                The forward pass is used to find events, i.e. change-point intervals, and the backward pass is used
                to improve the segmentation of the input signal into steady-state and event intervals.

                1. Forward pass:
                        For a given input vector X with length n o the following:
                            1. Take the next sample x_n+1 and append it to X
                            2. Update the clustering and the clustering structure, using the DBSCAN Algorithm
                            By doing this we get clusters C1 and C2 and a possible event_interval_t
                            3. Compute the loss for the given cluster and the given event model (i.e. Model 3)
                               The loss for a signal segment X and two clusters C1 and C2 is defined as follows,
                               it counts the number of samples that need to be corrected in order for the segment
                               to match the event model:
                               L(X) = a + b + c with
                               a: number of samples n from C2 with n <= u, with u being the lower bound of
                               the event_interval_t
                               b: number of samples n from C1 with n >= v, with v being the upper bound of
                               the  event_interval_t
                               c: number of samples n between u < n < v, so number of samples n in the event_interval_t that
                               belong to C1 or C2, i.e. to the stationary signal.

                               We compute this loss for all cluster combinations, i.e. if the event model checks are
                               passed by three (non noise) clusters, then we compute two loss values

                            4. Check if L(X) <= loss_thresh.
                            If not go to step 1. and take the next sample.

                            Note: we have included a savety mechanism to prevent memory errors: if the size of X is bigger
                            then future_window_size_n times of the original window size, then we return that
                            no event was detected and the user should continue with the next input as described in the
                            input section of the documentation.

                            If yes:
                               if multiple cluster combinations have passed the loss_thresh, then declare
                               an event as detected  detected, with the change-interval event_interval_t
                               that results from the cluster combination with the lowest loss L(X)
                               Go to step 5 and start the backward pass.

                2. Backward pass:
                            1. Delete the oldest sample x1 from the segment (i.e the first sample in X)
                            2. Update the clustering and the clustering structure, using the DBSCAN Algorithm
                            3. Check the loss L(X) for the detected segment and the detected event_interval_t.
                            If L(X) <= loss_thresh, go to step 2 again.
                            If L(X=) >= loss_thresh, i.e. if without the removed sample no event is detected anymore
                            reinsert the last-sample again and declare the segment X as a balanced event

                After this is done, the process restarts from the first sample of C2 (x_v).

                The whole algorithm is window-based, with a initial window size of window_size_n
                The event detector has the following hyperparameters that can be fine-tuned.
                    - DBSCAN Epsilon (dbscan_eps)
                    - DBSCAN Mininum Points (dbscan_min_pts)
                    - Window Size (window_size_n)
                    - Threshold for the Loss-Function (loss_thresh)
                    - Temporal Locality Epsilon (temp_eps)

                Non algorithm related parameters that can be adapted are:
                    - Number of datapoints after the input window that are fed to the algorithm future_window_size_n
                    - Flag to turn input window order checks on or of perform_input_order_checks
                    (see also the details in the input section of this documentation for the two parameters above)

                If you want to debug the inner workings of the algorithm, i.e. get plots on the clustering etc. .
                then initialize the estimator with debugging_mode=False.
                Using this in a graphical environment, like a jupyter notebook is highly recommended.
                This provides a tool to understand the inner workings of the algorithm in detail.
           """

    def __init__(self, dbscan_eps=0.05, dbscan_min_pts=3, window_size_n=5,
                 loss_thresh=40, temp_eps=0.8, debugging_mode=False, dbscan_multiprocessing=False, network_frequency=50,
                 values_per_second=2,**kwargs):
        """


        Args:
            dbscan_eps (float): Epsilon Parameter for the DBSCAN algorithm
            dbscan_min_pts (int): Minimum Points Parameter for the DBSCAN algorithm
            window_size_n (int): Window Size
            loss_thresh (int): treshhold fopr the loss-function
            temp_eps (float):  temporal locality epsilon
            perform_input_order_checks: check the correct input order before processing the input, as described in
            the doc-string
            debugging_mode (bool): activate if plots of the dbscan clustering shall be shown
            grid_search_mode (bool): activate to adapt the score function, if you want to perfrom grid-search
            values_per_second (int): values per second that are fed to the algorithm
            dbscan_multiprocessing (bool): default=False, if set to true multiple processes are used in the dbscan algorithm.
            If the Barsim_Sequential event detector is used within a multiprocessing environment, turning the dbscan_multiprocessing
            paramter to True, results in warnings by sklearn and the multiprocessing library, as no additional subprocesses can
            be spawned by the processes.


        """


        self.dbscan_eps = dbscan_eps
        self.dbscan_min_pts = dbscan_min_pts
        self.window_size_n = window_size_n
        self.loss_thresh = loss_thresh
        self.temp_eps = temp_eps

        self.values_per_second=values_per_second
        self.network_frequency = network_frequency # periods per second

        # initialize the corresponding parameter for the checks
        self.order_safety_check = None



        self.debugging_mode = debugging_mode

        self.dbscan_multiprocessing=dbscan_multiprocessing


    def fit(self):
        """
        Call before calling the predict function. Needed by the sklearn API.

        """

        self.is_fitted = True

    def predict(self, X):
        """
        Predict if the input provided contains an event or not.
        The input provided should be computed by the compute_input_signal function of this class.
        Args:
            X_overall (ndarray): Input computed by compute_input_signal function.

        Returns:
            event_interval_indices (tuple): (start_index, end_index), if no event detected None is returned

        """
        # Check if fit was called before
        check_is_fitted(self, ["is_fitted"])

        # 1. Check the input
        # 1.1 Perform general tests on the input array

        utils.assert_all_finite(X)
        X = utils.as_float_array(X)

        if self.debugging_mode == True:

            processing_start_time = time.process_time()

        # 2. Event Detection Logic

        # 2.1 Forward Pass


        event_detected = False # Flag to indicate if an event was detected or not

        # 2.1.2 Update the clustering and the clustering structure, using the DBSCAN Algorithm
        # By doing this we get clusters C1 and C2
        self._update_clustering(X)

        # Now check the mode constraints
        # Possible intervals event_interval_t are computed in the _check_event_model_constraints() function.
        checked_clusters = self._check_event_model_constraints()

        # If there are no clusters that pass the model constraint tests, the method _check_event_model_constraints()
        # returns None, else a list of triples (c1, c2, event_interval_t).

        if checked_clusters is None:
            return None #add the next new_datapoint. We go back to step 1 of the forward pass.

        # 2.1.3 Compute the Loss-values

        else: # at least one possible combination of two clusters fullfills the event model constraints-
            # Hence, we can proceed with step 3 of the forward pass.
            # Therefore, we compute the loss for the given cluster combination.
            # The formula for the loss, is explained in the doc-string in step 3 of the forward pass.

            event_cluster_combination = self._compute_and_evaluate_loss(checked_clusters)
            self.forward_clustering_structure = self.clustering_structure #save the forward clustering structure

            if event_cluster_combination is not None: #  event detected
                event_detected = True
                #leave the loop of adding new samples each round and continue the code after the loop

            else: # go back to step 1 and add the next sample
                return None

        if event_detected == True: #an event was detected in the forward pass, so the backward pass is started
            if self.debugging_mode == True:
                print("Event Detected at: " + str(event_cluster_combination))
                print("")
                print("")
                print(60*"*")
                print("Backward pass is starting")
                print(60 * "*")

            # Initialize the backward pass clustering with the forward pass clustering, in case already the
            # first sample that is removed, causes the algorithm to fail. Then the result from the forward
            # pass is the most balanced event
            self.backward_clustering_structure = self.forward_clustering_structure
            event_cluster_combination_balanced = event_cluster_combination

            # 2.2.1. Delete the oldest sample x1 from the segment (i.e the first sample in X)
            for i in range(1, len(X)-1):
                X_cut = X[i:] #delete the first i elements, i.e. in each round the oldest sample is removed

                # 2.2.2 Update the clustering structure
                self._update_clustering(X_cut) #the clustering_structure is overwritten, but the winning one
                # from the forward pass is still saved in the forward_clustering_structure attribute

                # 2.2.3 Compute the loss-for all clusters that are detected (except the detected)
                # Hence, we need to check the event model constraints again
                checked_clusters = self._check_event_model_constraints()

                if checked_clusters is None: #roleback with break
                    status = "break"
                    event_cluster_combination_balanced = self._roleback_backward_pass(status, event_cluster_combination_balanced,i)
                    break #finished

                else: #compute the loss
                    # 2.2.4 Check the loss-values for the detected segment
                    event_cluster_combination_below_loss = self._compute_and_evaluate_loss(checked_clusters)

                    if event_cluster_combination_below_loss is None: #roleback with break
                        status = "break"
                        event_cluster_combination_balanced = self._roleback_backward_pass(status,
                                                                                          event_cluster_combination_balanced,
                                                                                          i)
                        break #finished
                    else:
                        status = "continue"
                        event_cluster_combination_balanced = self._roleback_backward_pass(status,
                                                                                          event_cluster_combination_balanced,
                                                                                          i,
                                                                                          event_cluster_combination_below_loss
                                                                                          )
                        continue #not finished, next round, fiight

            event_start = event_cluster_combination_balanced[2][0]
            event_end = event_cluster_combination_balanced[2][-1]
            if self.debugging_mode == True:
                print("Balanced event detected in the Backward pass from " + str(event_start) + " to " + str(event_end))
            # In case an event is detected, the first sample of the second steady state segment (c2) should be fed to
            # the estimator again for further event detection, as described in the documentation.
            # We use the first 10 start values to perform the necessary input check if the corresponding parameter
            # perform_input_order_checks = True


            self.order_safety_check = {"first_10_start_values" : X[event_end + 1: event_end + 11] }

            if self.debugging_mode is True:
                elapsed_time = time.process_time() - processing_start_time
                print("")
                print("Processing this window took: " + str(elapsed_time) + " seconds")



            return (event_start, event_end)
        else:
            # also for the input order check
            # in case no event is detected, the user should feed back the last window_size_n samples of X.
            # this is implemented that way to prevent memory issues
            self.order_safety_check = {"first_10_start_values": X[-self.window_size_n:][:10]}

            if self.debugging_mode is True:
                elapsed_time = time.process_time() - processing_start_time
                print("")
                print("Processing this window took: " + str(elapsed_time) + " seconds")

            return None

    def _compute_and_evaluate_loss(self, checked_clusters):
        """
        Function to compute the loss values of the different cluster combinations.
        The formula for the loss, is explained in the doc-string in step 3 of the forward pass.

        Args:
            checked_clusters (list): of triples (c1, c2, event_interval_t)

        Returns:
            event_cluster_combination (tuple): triple of the winning cluster combination

        """

        if self.debugging_mode is True:
            print("")
            print("")
            print("Compute the Loss values for all cluster combinations that have passed the model constraints")
            print("They have to be smaller than: " + str(self.loss_thresh))

        event_model_loss_list = []
        for c1, c2, event_interval_t in checked_clusters:
            lower_event_bound_u = event_interval_t[0] - 1  # the interval starts at u + 1
            upper_event_bound_v = event_interval_t[-1] + 1  # the interval ends at v -1
            c1_indices = self.clustering_structure[c1]["Member_Indices"]
            c2_indices = self.clustering_structure[c2]["Member_Indices"]
            a = len(np.where(c2_indices <= lower_event_bound_u)[0]) # number of samples from c2 smaller than lower bound of event

            b = len(np.where(c1_indices >= upper_event_bound_v)[0]) # number of samples from c1 greater than upper bound of event

            c1_and_c2_indices = np.concatenate([c1_indices, c2_indices])

            # number of samples n between u < n < v, so number of samples n in the event_interval_t that
            # belong to C1 or C2, i.e. to the stationary signal.
            c = len(np.where((c1_and_c2_indices > lower_event_bound_u) & (c1_and_c2_indices < upper_event_bound_v))[0])


            event_model_loss = a + b + c

            event_model_loss_list.append(event_model_loss)

            if self.debugging_mode is True:
                print("\tLoss for clusters " + str(c1) + " and " + str(c2) + ": " + str(event_model_loss))
                print("\t\tComposited of: " + "a=" + str(a) + " b=" + str(b) + " c=" +str(c))

        # 2.1.4 Compare loss value to the threshold on the loss loss_thresh
        # We select the cluster combination with the smallest loss, that is below the threshold

        # Select the smallest loss value
        min_loss_idx = np.argmin(event_model_loss_list)  # delivers the index of the element with the smallest loss

        # Compare with the loss threshold, i.e. if the smallest loss is not smaller than the treshold, no other
        # loss will be in the array

        if event_model_loss_list[min_loss_idx] <= self.loss_thresh:  # if smaller than the threshold event detected
            event_cluster_combination = checked_clusters[min_loss_idx]  # get the winning event cluster combination
            if self.debugging_mode is True:
                print("\tEvent Cluster Combination determined")
            return event_cluster_combination

        else:
            return None

    def _update_clustering(self, X):
        """
        Using the DBSCAN Algorithm to update the clustering structure.
        All available CPUs are used to do so.
        Furthermore all relevant metrics are directly computed from the clustering result.

        The method sets the clustering_structure attribute of the estimator class:
            clustering_structure (dict): resulting nested clustering structure. contains the following keys
            For each cluster it contains: {"Cluster_Number" : {"Member_Indices": []"u" : int,"v" : int,"Loc" : float} }
            u and v are the smallest and biggest index of each cluster_i respectively.
            Loc is the temporal locality metric of each cluster_i.


        Args:
            X (ndarray): input window, shape=(n_samples, 2)

        Returns:
            None


        """



        # Do the clustering
        # Use all CPU's for this, i.e. set n_jobs = -1
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts, n_jobs=-1).fit(X)

        # Get the cluster labels for each datapoint in X
        X_cluster_labels = np.array(dbscan.labels_)

        # Noise samples get the "-1" class --> those are usually the transients
        # Get all unique cluster identifiers
        cluster_labels = np.unique(X_cluster_labels)


        if self.debugging_mode == True: #if in debuggin mode, plot the clusters

            if self.original_non_log is False:
                log_label = "(Log-Scale)"
            else:
                log_label = ""

            plt.clf()
            plt.scatter(x=np.arange(len(X)),y=X[:, 0], c=X_cluster_labels, cmap='Paired')
            plt.ylabel("Active Power " + log_label)
            plt.xlabel("Samples")
            plt.title("Clustering")
            plt.show()
            plt.clf()
            plt.scatter(x=X[:, 1], y=X[:,0],  c=X_cluster_labels, cmap='Paired')
            plt.ylabel("Active Power " + log_label)
            plt.xlabel("Reactive Power " + log_label)
            plt.title("Clustering")
            plt.show()

        clustering_structure = {}

        #build the cluster structure, for each cluster store the indices of the points.
        for cluster_i in cluster_labels:
            cluster_i_structure = {} #all the relvant information about cluster_i

            # Which datapoints (indices) belong to cluster_i
            cluster_i_member_indices = np.where(X_cluster_labels == cluster_i)[0]
            cluster_i_structure["Member_Indices"] = np.array(cluster_i_member_indices)

            # Determine u and v of the cluster (the timely first and last element, i.e. the min and max index)
            u = np.min(cluster_i_member_indices)
            v = np.max(cluster_i_member_indices)
            cluster_i_structure["u"] = u
            cluster_i_structure["v"] = v

            # compute the temporal locality of cluster_ci
            Loc_cluster_i = len(cluster_i_member_indices) / (v - u + 1) # len(cluster_i_member_indices) = n_samples_in_Ci
            cluster_i_structure["Loc"] = Loc_cluster_i

            # insert the structure of cluster_i into the overall clustering_structure
            clustering_structure[cluster_i] = cluster_i_structure


        self.clustering_structure = clustering_structure

        return None

    def _roleback_backward_pass(self, status, event_cluster_combination_balanced, i, event_cluster_combination_below_loss=None):
        """
        When the backward pass is performed, the oldest datapoint is removed in each iteration.
        After that, first the model constraints are evaluated.
        If they are violated, we roleback to the previous version by adding the oldest datapoint again
        and we are finished.
        In case the model constraints still hold, we recompute the loss.
        If the loss exceeds the threshold, we ne to roleback to the last version too.

        This roleback happens at to positions in the code (i.e. after the model constraints are evaluated and after
        the loss computation). Therefore, it is encapsulated in this function.

        Args:
            status (string): either "continue" or "break"
            i: current iteration index of the datapoint
            event_cluster_combination_balanced:
            event_cluster_combination_below_loss:

        Returns:

        """
        if status == "break":
            # if the loss is above the threshold
            # without the recently removed sample, take the previous combination and declare it as an
            # balanced event.
            # the previous clustering and the previous event_cluster_combination are saved from the previous
            # run automatically, so there is no need to perform the clustering again.

            # Attention: the event_interval indices are now matched to X_cut.
            # We want them to match the original input X instead.
            # Therefore we need to add + (i-1) to the indices, the  -1 is done because we take
            # the clustering and the state of X_cut from the previous, i.e. i-1, round.
            # This is the last round where the loss, was below the threshold, so it is still fine
            event_cluster_combination_balanced = list(event_cluster_combination_balanced)
            event_cluster_combination_balanced[2] = event_cluster_combination_balanced[2] + i  # the event_interval_t
            event_cluster_combination_balanced = tuple(event_cluster_combination_balanced)


            # The same is to be done for all the final cluster
            # The structure stored in self.backward_clustering_structure is valid, it is from the previous iteration
            for cluster_i, cluster_i_structure in self.backward_clustering_structure.items():
                cluster_i_structure["Member_Indices"] = cluster_i_structure["Member_Indices"] + int(i - 1)
                cluster_i_structure["u"] = cluster_i_structure["u"] + int(i - 1)
                cluster_i_structure["v"] = cluster_i_structure["v"] + int(i - 1)

                # Only the "Loc" is not updated (stays the same, regardless of the indexing)
                self.backward_clustering_structure[cluster_i] = cluster_i_structure

            return event_cluster_combination_balanced

        elif status == "continue":  # continue with the backward pass
            # update the backward_clustering_structure with the latest valid one
            # i.e. the new clustering structure

            self.backward_clustering_structure = self.clustering_structure
            event_cluster_combination_balanced = event_cluster_combination_below_loss  # same here
            return event_cluster_combination_balanced

        else:
            raise ValueError("Status code does not exist")

    def _check_event_model_constraints(self):
        """
        Checks the constraints the event model, i.e. event model 3, opposes on the input data.
        It uses the clustering_structure attribute, that is set in the _update_clustering() function.

        Arguments:

        Returns:
            checked_clusters (list): list of triples (c1, c2, event_interval_t)
                                    with c1 being the identifier of the first cluster, c2 the second cluster
                                    in the c1 - c2 cluster-combination, that have passed the model
                                    checks. The event_interval_t are the indices of the datapoints in between the two
                                    clusters.


        """

        if self.debugging_mode is True:
            print("")
            print("Perform check 1 to find non noise cluster")

        # (1) it contains at least two clusters C1 and C2, besides the outlier cluster, and the outlier Cluster C0
        # can be non empty. (The noisy samples are given the the cluster -1 in this implementation of DBSCAN)
        n_with_noise_clusters = len(self.clustering_structure)
        n_non_noise_clusters = n_with_noise_clusters - 1 if -1 in self.clustering_structure.keys() else n_with_noise_clusters

        if self.debugging_mode is True:
            print("Number of non noise_clusters: " + str(n_non_noise_clusters))

        if n_non_noise_clusters < 2: #check (1) not passed
            return None

        # If check (1) is passed, continue with check (2)

        # (2) clusters C1 and C2 have a high temporal locality, i.e. Loc(Ci) >= 1 - temp_eps
        # i.e. there are at least two, non noise, clusters with a high temporal locality

        check_two_clustering = {} #store the clusters that pass the test in a new structure

        if self.debugging_mode is True:
            print("")
            print("Perform check 2 with temp locality greater than " + str(1 - self.temp_eps))
            print("Cluster | Temporal Locality")
            print("--------|----------------- ")
        for cluster_i, cluster_i_structure in self.clustering_structure.items():
            if cluster_i != -1: # for all non noise clusters

                if self.debugging_mode is True:
                   print(str(cluster_i) + "       | " + str(cluster_i_structure["Loc"]))

                if cluster_i_structure["Loc"] >= 1 - self.temp_eps: #the central condition of condition (2)
                    check_two_clustering[cluster_i] = cluster_i_structure


        if self.debugging_mode is True:
            print("Number of clusters that pass temporal locality epsilon(Check 2): " + str(n_non_noise_clusters) + " (min 2 clusters) ")

        if len(check_two_clustering) < 2:  #check (2) not passed
            return None

        # (3) two clusters C1 and C2 do not interleave in the time domain.
        # There is a point s in C1 for which all points n > s do not belong to C1 anymore.
        # There is also a point i in C2 for which all points n < i do not belong to C2 anymore.
        # i.e. the maximum index s of C1 has to be smaller then the minimum index of C2
        checked_clusters = []

        # We need to compare all pairs of clusters in order to find all pairs that fullfill condition (3)

        if self.debugging_mode is True:
            print("")
            print("Perform check 3 for all combinations ")

        cluster_combinations = itertools.combinations(check_two_clustering, 2) #get all combinations, without replacement

        for cluster_i, cluster_j in cluster_combinations: #for each combinations

            # the cluster with the smaller u, occurs first in time, we name it C1 according to the paper terminology here
            if check_two_clustering[cluster_i]["u"] < check_two_clustering[cluster_j]["u"]:
                #  #cluster_i is starting before cluster_j
                c1 = cluster_i
                c2 = cluster_j
            else: #then cluster_j is starting before cluster i
                c1 = cluster_j
                c2 = cluster_i

            # now we check if they are overlapping
            # the maximum index of C1 has to be smaller then the minimum index of C2, then no point of C2 is in C1
            # and the other way round, i.e all points in C1 have to be smaller then u of C2

            if check_two_clustering[c1]["v"] < check_two_clustering[c2]["u"]: #no overlap detected
                # if thee clusters pass this check, we can compute the possible event_interval_t (i.e. X_t)
                # for them. This interval possibly contains an event and is made up of all points between cluster c1
                # and cluster c2 that are noisy-datapoints, i.e. that are within the -1 cluster. THe noisy points
                # have to lie between the upper-bound of c1 and the lower-bound of c2
                if -1 not in self.clustering_structure.keys():
                    return None
                else:
                    c0_indices = self.clustering_structure[-1]["Member_Indices"] #indices in the -1 (i.e. noise) cluster


                #ASSUMPTION if there is no noise cluster, then the check is not passed.
                # No event segment is between the two steady state clusters then.
                # Any other proceeding would cause problems in all subsequent steps too.

                if self.debugging_mode is True:
                    print("No overlap between cluster " + str(c1) + " and " + str(c2) + " (i.e. a good Candidate)")
                    print("\tPotential event window (noise cluster indices:  " + str(c0_indices))
                    print("\tCluster 1 v: " + str(check_two_clustering[c1]["v"] ))
                    print("\tCluster 2 u: " + str(check_two_clustering[c2]["u"]))

                # check the condition, no overlap between the noise and steady state clusters allowed: u is lower, v upper bound
                condition = ((c0_indices > check_two_clustering[c1]["v"]) & (c0_indices < check_two_clustering[c2]["u"]))
                event_interval_t = c0_indices[condition]

                if self.debugging_mode is True:
                    print("\tEvent Intervall between cluster " + str(c1) + " and " + str(c2) + " with indices " + str(event_interval_t))

                # If the event_interval_t contains no points, we do not add it to the list too,
                # i.e. this combinations does not contain a distinct event interval between the two steady state
                # cluster sections.
                if len(event_interval_t) != 0:

                    checked_clusters.append((c1, c2, event_interval_t))

                else:
                    if self.debugging_mode is True:
                        print("Check 3: Event interval between the steady states was empty")
        # now all non interleaving clusters are in the check_three_clustering structure
        # If there are still at least two non-noise clusters in check_three_clustering, then check (3) is passed,
        # else not.


        if self.debugging_mode is True:
            print("Number of cluster-pairs that pass Check 3: " + str(len(checked_clusters)) + " (min 1 pair)")

        if len(checked_clusters) < 1:  # check (3) not passed


            return None


        return checked_clusters

    def compute_input_signal(self, voltage, current, period_length, original_non_log=False, single_sample_mode=False):
        """
        This functions uses the instantaneous voltage and current signals to compute the real (P) and reactive power (Q).
        The period_length has to divide length of the input signals evenly, i.e. no remainder.
        If this is not the case, an exception is raised.
        This is because the features are computed on a period-based approximation.

        The features are computed per 0.5 seconds, thus, two features per second are computed.

        ASSUMPTION: To reduce the variation in the input for the clustering, we logarthmize the data as the
        authors have done in other publications by them. In this paper, they do not detail the pre-processing and
        input releated hyperparameters like the window size.
        If you do not agree with this assumption, set the original_non_log parameter to True, and the signal
        will not be converted to log-scale.

        Args:
            voltage (ndarray): one-dimensional voltage array
            current (ndarray: one-dimensional current array
            period_length (int): length of a period for the given dataset.
                           example: sampling_rate=10kHz, 50Hz basefrequency,
                           10kHz / 50 = 200 samples / period
            original_non_log (bool): default: False, if set to true, the non-logarithmized data is returned.
            single_sample_mode (bool): default: False, if set to true, the input checks are adapted for single sampels

        Returns:
            X (ndarray): feature vector with active and reactive power, with shape(window_size_n,2).
                         The first component at time t is the active, the second one the reactive power.
                         The active and reactive power are per values per second.



        """

        voltage, current = utils.check_X_y(voltage, current, force_all_finite=True, ensure_2d=False,allow_nd=False, y_numeric=True,
                  estimator="EventDet_Barsim_Sequential")

        period_length = int(period_length)

        compute_features_window = period_length*self.network_frequency/self.values_per_second #compute features averaged over this timeframe

        if single_sample_mode is False:
            # also ensure that the input length corresponds to full seconds

            if len(voltage) % (int(period_length) * self.network_frequency) != 0:
                raise ValueError("Ensure that the input signal can be equally divided with the period_length and full seconds! "
                                 "The input has to correspond to full seconds, e.g. lengths such as 1.3 seconds are not permitted")
        else:
            if len(voltage) % int(period_length) != 0:
                raise ValueError("Ensure that the input signal can be equally divided with the period_length and full seconds! "
                                 "The input has to correspond to full seconds, e.g. lengths such as 1.3 seconds are not permitted")

        self.original_non_log = original_non_log

        #Compute the active and reactive power using the Metrics class provided with this estimator
        Metrics = Test_Utility.Electrical_Metrics()

        active_power_P= Metrics.active_power(instant_voltage=voltage, instant_current=current,
                                            period_length=compute_features_window) #values per second

        apparent_power_S = Metrics.apparent_power(instant_voltage=voltage, instant_current=current,
                                            period_length=compute_features_window) #values per second


        reactive_power_Q = Metrics.reactive_power(apparent_power=apparent_power_S, active_power=active_power_P)

        self.period_length = compute_features_window #used to convert the offsets back to the original data

        #Now we combine the two features into a signel vector of shape (window_size_n,2)
        X = np.stack([active_power_P, reactive_power_Q], axis=1)

        if original_non_log == False:

            X = np.log(X) #logarithmize the signal as described in the assumption

        return X

    def _convert_relative_offset(self, relative_offset, raw_period_length=None):
        """
        Convert the relative offset that is computed relative to the input of the algorithm, i.e. the return
        value of the compute_input_signal() function.

        This utility function can be used to adapt the offset back to the raw data.

        "To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method "


        Args:
            relative_offset (int): the offset that needs to be converted
            raw_period_length (int): length in samples of one period in the original (the target) raw data

        Returns:
            target_offset (int): offset relative to the raw (target) input
        """

        if raw_period_length is None:
            raw_period_length = self.period_length

        if raw_period_length is None:
            raise ValueError("To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method ")

        target_offset = relative_offset * raw_period_length

        return target_offset

    def _convert_index_to_timestamp(self, index, start_timestamp_of_window):
        """
        Function to convert an index that is relative to the start_timestamp of a window that was computed
        by the compute_input_signal function to a timestamp object.

        Args:
            index (int): index to convert, relative to the input window. Features that have been used to do the event
            detection and to get the index, have to be computed according to the compute_input_signal function.
            Otherwise the timestamps returned by this function can be wrong.

            start_timestamp_of_window(datetime): start timestamp of the window the event index is located in.
            network_frequency(int): basic network frequency
        Returns:

            event_timestamp (datetime)

        """

        seconds_since_start = index / self.values_per_second # 2 values per second
        event_timestamp = start_timestamp_of_window + timedelta(seconds=seconds_since_start)

        return event_timestamp

    @staticmethod
    def score(ground_truth_events_p, detected_events_p, number_of_samples_in_dataset_p=None,
              tolerance_limit_sec_p=1, return_event_lists_p=False):
        """
        This function can be used to determined the values of the confusion matrix.

        It works as follows:
        For each of the detected events, we build an interval around the detected timestamp using the tolerance_limit_sec
            Then we check for each of the detected intervals if we find a ground truth event that falls within this intervall
             If this is the case we have found a True Positive Event (TP) :
               1. TP += 1
               2. Eliminate this event from the ground truth list, in order to: avoid double detection
               In the case there is more then one ground truth events that fits into the interval, we take the first one
             If this is not the case we have found a False Positive Event (FP):
               1. FP += 1

             After this procedure: all events that are left in the ground truth list have not been detected by the event detection algorithm
             (because they were not removed from the in the previous step)
             Hence those events are actually true, but are delcared to be false by the algorithm: They are False Negatives (FN)
             For each of those events left in the ground truth list: FN += 1

             Finally we determine the number of True Negative Events(TN):
             Lastly, when all the detected and groundtruth events have been processed, the TN are calculated by
             subtracting the TP, FN, and FP from the number of samples in the dataset, i.e., all the positions where an event could have happened
             The start and the end time of the dataset and the information about sampling rate is used to compute the total
             number of possible event timestamps.

        Parameters
        ----------
        ground_truth_events_p: list
            List of datetime objects, containing the ground truth event timestamps.

        detected_events_p: list
            List of datetime objects, containing the event detected by the event detector

        number_of_samples_in_dataset_p: int
            Number of samples, i.e. number of possible event positions in the dataset.
            When computing period wise features, like RMS, values use the corresponding number of samples.
            The value is used to compute the number of True Negative Events.

        tolerance_limit_sec_p: int, optional (default = 1)
            The tolerance limit that is used to determine the scores of the confusion matrix.

        return_events_list_p: bool, optional (default = False)
            If set to True, the events that are TP, FP, and FN are returned in the results dictionary
        Returns
        -------

        if grid_search_mode is set to True, only the f1score is returned, otherwise:

        results: dictionary
            A dictionary containing the following keys:  tp, fp, fn, tn, f1score, recall, precision, fpp, fpr
            With being:
            tp = True Positives
            fp = False Positives
            fn = False Negatives
            tn = True Negatives
            f1score = F1-Score Metric
            recall = Recall Metric
            precision = Precision Metric
            fpp = False Positive Percentage Metric
            fpr = False Positive Rate Metric

            If return_events_list_p is set to True, the results also contain the following keys:
            tn_events = events that are tn
            fp_events = events that are fp
            fn_events = events that are fn

        """

        ground_truth_events = ground_truth_events_p #ground_truth_events_p.copy() #copy the events
        detected_events = detected_events_p.copy() #copy the events

        results = {}
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        TP_events = []
        FP_events = []
        FN_events = []


        # Compute the Tolerance Intervals around the detected events and iterate through the detected events
        for detected_event in detected_events:
            upper_limit = detected_event + timedelta(seconds=tolerance_limit_sec_p)
            lower_limit = detected_event - timedelta(seconds=tolerance_limit_sec_p)

            # Check which ground_truth events lie within this interval
            matching_ground_truth_events = [(index, ground_truth_event) for index, ground_truth_event in
                                            enumerate(ground_truth_events) if
                                            lower_limit <= ground_truth_event <= upper_limit]

            if len(matching_ground_truth_events) > 0:  # at least (minimum) one ground truth event was detected
                TP += 1
                first_index = matching_ground_truth_events[0][0]

                TP_events.append(ground_truth_events[first_index])
                # It can be possible that there are multiple ground truth events within this interval, hence,
                # we only consider the first one and remove it from the list
                del ground_truth_events[first_index]

            else:  # No Matching Event was found
                FP_events.append(detected_event)
                FP += 1

        # Use the remaining ground truth events to determine the number of FN
        FN = len(ground_truth_events)
        FN_events = ground_truth_events.copy()

        # Use the number of samples in the dataset to determine the TNs in the dataset
        # For BLUED we have data from 20/10/2011 11:58:32.623499 to 27/10/2011 14:10:40.0

        TN = number_of_samples_in_dataset_p - (TP + FP + FN)
        # Compute the final scores
        fpp = FP / (TP + FN)
        fpr = FP / (FP + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * precision * recall) / (precision + recall)

        results["tp"] = TP
        results["fp"] = FP
        results["fn"] = FN
        results["tn"] = TN
        results["f1score"] = f1score
        results["recall"] = recall
        results["precision"] = precision
        results["fpp"] = fpp
        results["fpr"] = fpr

        if return_event_lists_p is True:
            results["tp_events"] = TP_events
            results["fp_events"] = FP_events
            results["fn_events"] = FN_events

        return results