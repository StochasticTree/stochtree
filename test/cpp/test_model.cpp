/*!
 * End-to-end test of the "grow-from-root" procedure, removing stochastic aspects:
 *   1. Data is fixed so that log-likelihoods can be computed deterministically
 *   2. Variance parameters are fixed
 *   3. Leaf node parameters are set to the posterior mean, rather than sampled
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/model.h>
#include <stochtree/model_draw.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>
#include <vector>

class ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Define a common dataset to be used in multiple tests
    n = 40;
    p = 3;
    std::vector<double> data_vector = {
      554.5587, 0.113703411, 0.55333359, 0.92640048, 
      652.06658, 0.622299405, 0.64640609, 0.47190972, 
      314.83592, 0.609274733, 0.31182431, 0.14261534, 
      624.26646, 0.623379442, 0.6218192, 0.54426976, 
      334.06715, 0.860915384, 0.32977018, 0.19617465, 
      506.97611, 0.640310605, 0.50199747, 0.89858049, 
      676.0034, 0.009495756, 0.67709453, 0.38949978, 
      487.52182, 0.232550506, 0.48499124, 0.31087078, 
      248.58881, 0.666083758, 0.24392883, 0.16002866, 
      768.36752, 0.514251141, 0.76545979, 0.89618585, 
      77.25473, 0.693591292, 0.07377988, 0.16639378, 
      311.95601, 0.544974836, 0.3096866, 0.9004246, 
      718.31889, 0.282733584, 0.71727174, 0.1340782, 
      509.81137, 0.923433484, 0.50454591, 0.13161413, 
      156.53081, 0.29231584, 0.15299896, 0.1052875, 
      507.96657, 0.837295628, 0.50393349, 0.51158358, 
      494.00134, 0.286223285, 0.49396092, 0.30019905, 
      751.81072, 0.26682078, 0.7512002, 0.0267169, 
      175.8417, 0.18672279, 0.17464982, 0.30964743, 
      849.23648, 0.232225911, 0.84839241, 0.74211966, 
      866.2391, 0.316612455, 0.86483383, 0.03545673, 
      43.20075, 0.302693371, 0.04185728, 0.56507611, 
      316.60508, 0.159046003, 0.31718216, 0.28025778, 
      13.77613, 0.039995918, 0.01374994, 0.20419632, 
      240.96996, 0.218799541, 0.23902573, 0.1337389, 
      711.24522, 0.810598552, 0.70649462, 0.32568192, 
      311.27324, 0.525697547, 0.30809476, 0.15506197, 
      512.71812, 0.914658166, 0.50854757, 0.12996214, 
      55.61175, 0.831345047, 0.05164662, 0.43553106, 
      563.60416, 0.045770263, 0.56456984, 0.03864265, 
      123.70749, 0.456091482, 0.12148019, 0.71330156, 
      894.41751, 0.265186672, 0.89283638, 0.10076904, 
      17.85658, 0.304672203, 0.01462726, 0.95030494, 
      786.65915, 0.50730687, 0.7831211, 0.12181776, 
      90.37123, 0.181096208, 0.08996133, 0.21965662, 
      523.34388, 0.759670635, 0.51918998, 0.91308777, 
      384.13832, 0.201248038, 0.38426669, 0.94585312, 
      72.22475, 0.258809819, 0.0700525, 0.27915622, 
      326.57809, 0.992150418, 0.32064442, 0.12347109, 
      674.65328, 0.80735234, 0.6684954, 0.79716046, 
    };
    y_bar= 431.229372;
    
    // Define any config parameters that aren't defaults
    const char* params = "label_column=0 num_trees=2 min_data_in_leaf=1 alpha=0.95 beta=1.25";
    auto param = StochTree::Config::Str2Map(params);
    config.Set(param);

    // Define data loader
    StochTree::DataLoader dataset_loader(config, 1, nullptr);

    // Load some test data
    dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));
  }

  // void TearDown() override {}
  std::unique_ptr<StochTree::Dataset> dataset;
  std::vector<std::vector<StochTree::data_size_t>> tree_observation_indices;
  StochTree::Config config;
  double y_bar;
  int p;
  StochTree::data_size_t n;
};

