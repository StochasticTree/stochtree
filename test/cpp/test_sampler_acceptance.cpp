#include <gtest/gtest.h>
#include <stochtree/tree_sampler.h>
#include <numeric>

namespace {
struct ProposalFixture {
  StochTree::ForestDataset data;
  std::vector<StochTree::FeatureType> types{StochTree::FeatureType::kNumeric};
  StochTree::Tree tree;
  std::unique_ptr<StochTree::ForestTracker> tracker;
  StochTree::TreePrior prior{0.6, 2., 2};
  std::mt19937 gen{19};
  ProposalFixture(int n) {
    std::vector<double> x(n); std::iota(x.begin(), x.end(), 0.);
    data.AddCovariates(x.data(), n, 1, true);
    tracker = std::make_unique<StochTree::ForestTracker>(data.GetCovariates(), types, 1, n);
    tree.Init(1);
  }
  auto State() { return StochTree::MCMCGetProposalState(&tree, *tracker, 0, prior.GetMinSamplesLeaf()); }
  void Grow(int leaf, double cut) {
    StochTree::TreeSplit split(cut);
    StochTree::AddSplitToModel(*tracker, data, prior, split, gen, &tree, 0, leaf, 0);
  }
};
void Same(const StochTree::MCMCProposalState& a, const StochTree::MCMCProposalState& b) {
  EXPECT_EQ(a.leaves, b.leaves); EXPECT_EQ(a.leaf_parents, b.leaf_parents);
  EXPECT_EQ(a.grow_candidates, b.grow_candidates);
}
}

TEST(SamplerAcceptance, GrowThenPruneUsesActualWholeTreeState) {
  ProposalFixture f(12);
  for (auto proposal : std::vector<std::pair<int,double>>{{0,3.5},{2,7.5},{1,1.5}}) {
    int leaf=proposal.first;
    int n=f.tracker->UnsortedNodeSize(0,leaf);
    int left_n=0;
    for (auto i=f.tracker->UnsortedNodeBeginIterator(0,leaf);i!=f.tracker->UnsortedNodeEndIterator(0,leaf);++i)
      left_n += f.data.CovariateValue(*i,0)<=proposal.second;
    auto old=f.State();
    auto predicted=StochTree::MCMCStateAfterGrow(&f.tree,old,leaf,left_n,n-left_n,2);
    f.Grow(leaf,proposal.second);
    auto actual=f.State(); Same(predicted,actual);
    auto reverse=StochTree::MCMCStateAfterPrune(&f.tree,actual,leaf,left_n,n-left_n,2);
    Same(reverse,old);
    double q_forward=old.GrowProbability()/old.leaves;
    double q_reverse=actual.PruneProbability()/actual.leaf_parents;
    EXPECT_NEAR(StochTree::MCMCLogProposalRatio(old,actual,true),std::log(q_reverse/q_forward),1e-14);
    EXPECT_NEAR(StochTree::MCMCLogProposalRatio(actual,old,false),std::log(q_forward/q_reverse),1e-14);
  }
}

TEST(SamplerAcceptance, OtherLeavesAndLeafParentReplacementMatter) {
  ProposalFixture f(20); f.Grow(0,3.5);
  auto old=f.State();
  auto next=StochTree::MCMCStateAfterGrow(&f.tree,old,1,2,2,2);
  f.Grow(1,1.5); Same(next,f.State());
  EXPECT_EQ(next.leaf_parents,1); // root stops being a leaf parent
  EXPECT_EQ(next.grow_candidates,1); // the unaffected 16-observation leaf
  EXPECT_DOUBLE_EQ(next.PruneProbability(),0.5);
}

TEST(SamplerAcceptance, EqualityBoundaryAndPruningToRoot) {
  ProposalFixture f(4);
  auto root=f.State(); EXPECT_DOUBLE_EQ(root.GrowProbability(),1.);
  f.Grow(0,1.5); auto split=f.State();
  EXPECT_DOUBLE_EQ(split.PruneProbability(),1.);
  auto back=StochTree::MCMCStateAfterPrune(&f.tree,split,0,2,2,2);
  Same(back,root); EXPECT_DOUBLE_EQ(back.GrowProbability(),1.);
  EXPECT_DOUBLE_EQ(StochTree::MCMCLogProposalRatio(root,split,true),0.);
  EXPECT_DOUBLE_EQ(StochTree::MCMCLogProposalRatio(split,root,false),0.);
}

TEST(SamplerAcceptance, InactiveNodesDoNotPreventRootProbability) {
  ProposalFixture f(8); f.Grow(0,3.5);
  StochTree::RemoveSplitFromModel(*f.tracker,f.data,f.prior,f.gen,&f.tree,0,0,1,2);
  EXPECT_DOUBLE_EQ(f.State().GrowProbability(),1.);
  EXPECT_DOUBLE_EQ(f.State().PruneProbability(),0.);
}

TEST(SamplerAcceptance, UndersizedRootRetainsTree) {
  ProposalFixture f(3);
  std::vector<double> y(3,0.); StochTree::ColumnVector residual(y.data(),3);
  StochTree::GaussianConstantLeafModel model(1.);
  StochTree::ForestContainer forests(1); std::vector<double> weights{1.};
  StochTree::MCMCSampleTreeOneIter<StochTree::GaussianConstantLeafModel,StochTree::GaussianConstantSuffStat>(
    &f.tree,*f.tracker,forests,model,f.data,residual,f.prior,f.gen,weights,0,1.,1);
  EXPECT_EQ(f.tree.NumLeaves(),1);
}
