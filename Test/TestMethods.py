import unittest
import sys
sys.path.insert(0,'.') # Add this directory to path.
import misc
import HEX
import Datamanager
import network
import torch
#from misc import *

# * run python Test/TestMethods.py -v[Optional]

class TestMics(unittest.TestCase):
    def test_int_to_one_hot_vector(self):
        self.assertListEqual(misc.int_to_one_hot_vector(2,3),[0,0,1])
        self.assertListEqual(misc.int_to_one_hot_vector(1,2),[0,1])
    
    def test_int_to_one_hot_vector_rev(self):
        self.assertListEqual(misc.int_to_one_hot_vector_rev(0,2),[0,1])
        self.assertListEqual(misc.int_to_one_hot_vector_rev(1,2),[1,0])
        #self.assertListEqual(misc.int_to_one_hot_vector_rev(1,2),[0,1])
    
    def test_int_to_binary(self):
        self.assertListEqual(misc.int_to_binary(2,2),[1,0])
        self.assertListEqual(misc.int_to_binary(1,2),[0,1])
    
    def test_int_to_binary_rev(self):
        self.assertListEqual(misc.int_to_binary_rev(2,2),[0,1])
        self.assertListEqual(misc.int_to_binary_rev(1,2),[1,0])
        self.assertListEqual(misc.int_to_binary_rev(0,2),[0,0])
        with self.assertRaises(ValueError):
            misc.int_to_binary_rev(2,1)
        #self.assertRaises(ValueError, misc.int_to_binary(2,1)) # Doest work

    def test_normalize_array(self):
        self.assertAlmostEqual(misc.normalize_array([0,1,2,0]),[0,1/3,2/3,0])
        self.assertAlmostEqual(sum(misc.normalize_array([0,1,2,0])),1.00)
        self.assertListEqual(misc.normalize_array([0,1,200,100]),[0,1/301,200/301,100/301])
        self.assertAlmostEqual(sum(misc.normalize_array([0,1,200,100])),1.00)

class TestHEX(unittest.TestCase):
    def test_board_to_nn_input(self):
        hex = HEX.HEX(2) # Init game with 4 pieces
        self.assertListEqual(hex.state.get_state_as_input(),[0,0,0,0,0,0,0,0])
        # Play some turns.
        hex.play((0,0)) # ! Player 1 (1,0)
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,0,0,0,0,0])
        hex.play((0,1)) # ! Player 2 (0,1)
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,1,0,0,0,0])
        hex.play((1,1)) # ! Player 1 (1,0)
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,1,0,0,1,0])
        hex.play((1,0)) # ! Player 2 (1,0)
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,1,0,1,1,0])

        hex = HEX.HEX(3) # Init game with 9 pieces : 18 states
        hex.play((0,0)) # ! Player 1 (1,0)                  [(00,(01,(02,(10,(11,(12,(20,(21,(22]
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        hex.play((0,1)) # ! Player 2 (0,1)
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        hex.play((1,1)) # ! Player 1 (1,0)
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        hex.play((1,0)) # ! Player 2 (0,1)
        self.assertListEqual(hex.state.get_state_as_input(),[1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0])

    """def test_board_legal_actions(self): # TODO: Fix legal_actions return np list.
        # Test wheter or not a play affects the legal_moves array
        hex = HEX.HEX(3)# 3x3 = 9
        self.assertListEqual(hex.get_legal_actions_bool(),[1,1,1,1,1,1,1,1,1])
        hex.play((0,0)) 
        self.assertListEqual(hex.get_legal_actions_bool(),[0,1,1,1,1,1,1,1,1])
        hex.play((0,1)) 
        self.assertListEqual(hex.get_legal_actions_bool(),[0,0,1,1,1,1,1,1,1])
        hex.play((1,0)) 
        self.assertListEqual(hex.get_legal_actions_bool(),[0,0,1,0,1,1,1,1,1])
        hex.play((1,1)) 
        self.assertListEqual(hex.get_legal_actions_bool(),[0,0,1,0,0,1,1,1,1])
        """
    
    def test_winning_condition(self):
        hex = HEX.HEX(5)
        hex.play((0,0))
        hex.play((3,3))
        hex.play((0,1))
        hex.play((0,2))
        hex.play((1,1))
        hex.play((1,2))
        hex.play((2,1))
        hex.play((2,3))
        hex.play((3,1))
        hex.play((4,1))
        hex.play((4,0))
        hex.play((1,3))
        hex.play((3,2))
        hex.play((0,3))
        hex.play((4,2))
        hex.play((0,4))
        hex.play((4,3))
        hex.play((1,4))
        hex.play((4,4))
        self.assertEqual(hex.is_game_over(),True)
        self.assertEqual(hex.get_winner(),1)

    def test_winning_condition_2(self):
        hex = HEX.HEX(5)
        hex.init_player_turn(2) # player 2 should start
        hex.play((0,0))
        hex.play((3,3))
        hex.play((0,1))
        hex.play((0,2))
        hex.play((1,1))
        hex.play((1,2))
        hex.play((2,1))
        hex.play((2,3))
        hex.play((3,1))
        hex.play((4,1))
        hex.play((4,0))
        hex.play((1,3))
        hex.play((3,2))
        hex.play((0,3))
        hex.play((4,2))
        hex.play((0,4))
        hex.play((4,3))
        hex.play((1,4))
        hex.play((4,4))
        self.assertEqual(hex.is_game_over(),True)
        self.assertEqual(hex.get_winner(),2)

class TestDatamanager(unittest.TestCase):
    def test_read_csv(self):
        datamanager = Datamanager.Datamanager("Data/test_dataset_update.csv")
        data = datamanager.read_csv()
        self.assertEqual(len(data),11)
    
    def test_write_csv(self):
        datamanager = Datamanager.Datamanager("Data/test_dataset_update.csv")
        data = datamanager.read_csv("Data/test_dataset_update.csv")

        data_values = [
            ["0,1,2,3,4,5,6,7,8,9,10","1,0","5,10,2,5,6,1,2,6"],
            ["1,2,3,4,5,6,7,8,9,0,1","0,1","7,9,1,57,8,2,1,2"],
            ["2,3,4,5,6,7,8,9,1,0","1,0","5,8,9,0,6,2,3,6,8,1"]]

        datamanager.update_csv_limit(data=data_values,header=["board","PID","board_target"],limit=10)
        data_updated = datamanager.read_csv("Data/test_dataset_update.csv")
        self.assertEqual(len(data_updated), len(data))

        datamanager_2 = Datamanager.Datamanager("Data/test_dataset_update_2.csv")
        data_2 = datamanager_2.read_csv()
        data_values = [
            ["x","x","x"],
            ["y","y","y"],
            ["z","z","z"]]
        
        datamanager_2.update_csv_limit(data=data_values,limit=10)
        data_updated_2 = datamanager_2.read_csv()
        self.assertEqual(len(data_updated_2), len(data_2))
    
    def test_return_batch(self):
        datamanager = Datamanager.Datamanager("Data/data_r_test.csv",dim=5)

        x,y = datamanager.return_batch(1)
        self.assertEqual(x.shape[0],1)
        self.assertEqual(y.shape[0],1)

    def test_get_length_dataset(self):
        datamanager = Datamanager.Datamanager("Data/test_dataset_update.csv")
        length = datamanager.get_buffer_size()
        self.assertEqual(length,11)

class TestNetwork(unittest.TestCase):
    # Input shape (N, 25)
    # sum need an dimention too

    def test_softmax(self):
        import torch.nn.functional as F
        #print("TestNetwork")
        inputs = torch.tensor([[1,2,3,4,5],[1,2,3,4,5]]).float()
        #print(F.softmax(inputs,dim=-1))
        inputs = F.softmax(inputs,dim=-1)
        #print(inputs)
        inputs = torch.sum(inputs,dim=-1)
        outputs = torch.tensor([1,1]).float()
        #print(inputs,inputs.shape,outputs,outputs.shape)
        #F.softmax()
        self.assertListEqual(list(inputs.numpy()), list(outputs.numpy()))

    def test_mse_loss(self):
        # Bad loss functions: 
        # cross_entropy_loss, nll_loss, multilabel_margin_loss, multi_margin_loss
        # hinge_embedding_loss
    
        import torch.nn.functional as F
        inputs = torch.tensor([[.1,.1,.1,.2,.5],[.4,.2,.1,.1,.2],[.5,.2,.1,.1,.1]]).float()
        targets = torch.tensor([[.1,.1,.1,.2,.5],[.5,.2,.1,.1,.1],[.5,.2,.1,.1,.1]]).float()
        mc_loss = network.MultiClassCrossEntropyLoss()
        cce_loss = network.CategoricalCrossEntropyLoss()
        rmse_loss = network.RootMeanSquareLoss()

        mse_r = F.mse_loss(inputs,targets)
        mse_r2 = F.mse_loss(inputs,targets,reduction='sum')
        mc_r = mc_loss(inputs, targets)
        cce_r = cce_loss(inputs, targets)
        l1_r = F.l1_loss(inputs,targets)
        l1a_r = F.l1_loss(inputs,targets,reduction='sum')
        pois_r = F.poisson_nll_loss(inputs,targets,log_input=False, full=True)
        rmse_r = rmse_loss(inputs, targets)
        print("\nResults mse:{:.5f} mse_s:{:.5f} mc:{:.5f} cce:{:.5f} l1:{:.5f} l1a_r:{:.5f} poi:{:.5f} rmse:{:.5f}"
            .format(mse_r,mse_r2,mc_r,cce_r,l1_r,l1a_r,pois_r, rmse_r))

        targets_w = torch.tensor([[.5,.0,.1,.0,.1],[.0,.0,.3,.6,.1],[.5,.2,.1,.1,.1]]).float()

        mse_r = F.mse_loss(inputs,targets_w)
        mse_r2 = F.mse_loss(inputs,targets_w, reduction='sum')
        mc_r = mc_loss(inputs, targets_w)
        cce_r = cce_loss(inputs, targets_w)
        l1_r = F.l1_loss(inputs,targets_w)
        l1a_r = F.l1_loss(inputs,targets_w,reduction='sum')
        pois_r = F.poisson_nll_loss(inputs,targets_w,log_input=False, full=True)
        rmse_r = rmse_loss(inputs,targets_w)
        print("\nResults mse:{:.5f} mse_s:{:.5f} mc:{:.5f} cce:{:.5f} l1:{:.5f} l1a_r:{:.5f} poi:{:.5f} rmse:{:.5f}"
            .format(mse_r,mse_r2,mc_r,cce_r,l1_r,l1a_r,pois_r, rmse_r))



if __name__=="__main__":
    unittest.main()