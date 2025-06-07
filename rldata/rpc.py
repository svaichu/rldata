
from rldata.base import *
from torch.distributed import rpc


class RPCRLData(RLData):
    def init_rpc(self, name="agent"):
        """
        Initialize the online buffer for RPC communication.
        This method is called to set up the buffer for remote procedure calls.
        
        Args:
            name (str): The name of the process (e.g., "agent" or "simulator").
        """
        # initialize rpc
        if name == "agent":
            self.rank = 0
        elif name == "simulator":
            self.rank = 1
        self.AGENT_RANK = 0
        self.SIMULATOR_RANK = 1
        rpc.init_rpc(name, rank=self.rank, world_size=2)
    
    def loadPolicy(self, policy):
        """
        Load the policy for the online buffer.
        
        Args:
            policy (BasePolicy): The policy to be used for the online buffer.
        """
        self.policy = policy
    
    def getAction(self, obs):
        """
        Get the action from the policy based on the current observations.
        
        Args:
            obs: Input for the model, typically a tensor or a dictionary of tensors.
        
        Returns:
            torch.Tensor: The action to be taken by the agent.
        """
        # check if policy is loaded
        if not hasattr(self, "policy"):
            raise RuntimeError("Policy is not loaded. Please load a policy before getting actions.")
        
        print(f"[INFO] Getting action for observation: {obs}")
        # get action from the policy
        action = self.policy.get_action(obs)
        return action
    
    def loadEnv(self, env):
        """
        Load the environment for the online buffer.
        
        Args:
            env (gym.Env): The environment to be used for the online buffer.
        """
        self.env = env
        # policy = None
    
    def remoteStep(self, action):
        """
        Perform a remote step
        """
        if self.rank != self.SIMULATOR_RANK:
            raise RuntimeError("This method can only be called from the simulator rank.")

        if not hasattr(self, "env"):
            raise RuntimeError("Environment is not loaded. Please load an environment before performing a remote step.")
        
        obs, _, dones, obs_dict = self.env.step(action)
    
    def remote_getAction(self, obs):
        """
        Get remote action based on the current observations.
        
        Args:
            obs: Input for the model, typically a tensor or a dictionary of tensors.
        
        Returns:
            torch.Tensor: The action to be taken by the agent.
        """
        if self.rank != self.SIMULATOR_RANK:
            raise RuntimeError("This method can only be called from the simulator rank.")

        action = rpc.rpc_sync("agent", self.getAction, args=(obs))
        return action    

    def shutdown_rpc(self):
        """
        Shutdown the RPC communication.
        
        This method is called to clean up the RPC communication and release resources.
        """
        rpc.shutdown()
        # dist.destroy_process_group()
    
        def remoteSimStep(self):
        """
        Perform a remote simulation step.
        
        This method is called to perform a simulation step in the remote environment.
        It can be used to update the state of the simulation and collect data.
        """
        
        # check this is called from the self.rank 0
        if self.rank != 0: 
            raise RuntimeError("This method can only be called from the agent rank (0).")
        # get the remote reference to the simulator
        simulator_rref = rpc.remote("simulator", Simulation, args=(self._modality_configs,))
        # call the remote step function
        future = self.remoteRun(OnlineBuffer.add_obs_from_isaac, simulator_rref)
        # wait for the future to complete
        processed_obs = future.wait()
        # add the processed observation to the replay buffer
        self.add_to_replay_buffer(processed_obs)
    
    # def shutdown(self):
    #     """
    #     Shutdown the online buffer.
        
    #     This method is called to clean up the online buffer and release resources.
    #     It should be called at the end of the simulation or when the buffer is no longer needed.
    #     """
        
    
    def getRemoteActions(self, obs):
        """
        Get remote actions based on the current observations.
        
        Args:
            obs (dict): The current observations from the environment.
        
        Returns:
            dict: The actions to be taken based on the observations.
        """
        # check this is CANNOT be called from the self.rank 0
        if self.rank == 0: 
            raise RuntimeError("This method CANNOT be called from the agent rank (0).")
        # get the remote reference to the simulator
        agent_rref = rpc.remote("simulator", OnlineBuffer, args=(self._modality_configs,))
        # call the remote step function
        future = self.remoteRun(OnlineBuffer.add_obs_from_isaac, agent_rref)
        # wait for the future to complete
        processed_obs = future.wait()
        return processed_obs
