#include <gtest/gtest.h>
#include "global_planner/tta_planner.hpp"

TEST(TTAPlanner, ComputeCenterline_StraightLine)
{
  TTAPlanner planner;

  std::vector<BoundaryPoint> left_boundary = {
    {0.0, 0.0},
    {0.0, 2.0},
    {0.0, 4.0}
  };

  std::vector<BoundaryPoint> right_boundary = {
    {2.0, 0.0},
    {2.0, 2.0},
    {2.0, 4.0}
  };

  std::vector<BoundaryPoint> centerline;

  ASSERT_TRUE(planner.computeCenterline(left_boundary, right_boundary, centerline));
  ASSERT_EQ(centerline.size(), 4u);

  EXPECT_DOUBLE_EQ(centerline[0].x, 1.0);
  EXPECT_DOUBLE_EQ(centerline[0].y, 0.0);

  EXPECT_DOUBLE_EQ(centerline[1].x, 1.0);
  EXPECT_DOUBLE_EQ(centerline[1].y, 2.0);

  EXPECT_DOUBLE_EQ(centerline[2].x, 1.0);
  EXPECT_DOUBLE_EQ(centerline[2].y, 4.0);

  EXPECT_DOUBLE_EQ(centerline[3].x, centerline[0].x);
  EXPECT_DOUBLE_EQ(centerline[3].y, centerline[0].y);
}

TEST(TTAPlanner, ComputeCenterline_Unordered)
{
  TTAPlanner planner;

  std::vector<BoundaryPoint> left_boundary = {
    {0.0, 4.0},
    {0.0, 0.0},
    {0.0, 2.0}
  };

  std::vector<BoundaryPoint> right_boundary = {
    {2.0, 2.0},
    {2.0, 4.0},
    {2.0, 0.0}
  };

  std::vector<BoundaryPoint> centerline;

  ASSERT_TRUE(planner.computeCenterline(left_boundary, right_boundary, centerline));
  ASSERT_EQ(centerline.size(), 4u);

  EXPECT_NEAR(centerline[0].x, 1.0, 1e-9);
  EXPECT_NEAR(centerline[1].x, 1.0, 1e-9);
  EXPECT_NEAR(centerline[2].x, 1.0, 1e-9);
}

TEST(TTAPlanner, ComputeCenterLine_RemovedDuplicates)
{
  TTAPlanner planner;

  std::vector<BoundaryPoint> left_boundary = {
    {0.0, 0.0},
    {0.0, 2.0},
    {0.0, 2.0}, // duplicate point
    {0.0, 4.0}
  };

  std::vector<BoundaryPoint> right_boundary = {
    {2.0, 0.0},
    {2.0, 2.0},
    {2.0, 4.0}
  };

  std::vector<BoundaryPoint> centerline;

  ASSERT_TRUE(planner.computeCenterline(left_boundary, right_boundary, centerline));
  EXPECT_EQ(centerline.size(), 4u);
}

TEST(TTAPlanner, ComputeCenterLine_TooFarApart)
{
  TTAPlanner planner;

  std::vector<BoundaryPoint> left_boundary = {
    {0.0, 0.0},
    {0.0, 2.0},
    {0.0, 4.0}
  };

  std::vector<BoundaryPoint> right_boundary = {
    {100.0, 0.0},
    {100.0, 2.0},
    {100.0, 4.0}
  };

  std::vector<BoundaryPoint> centerline;

  EXPECT_FALSE(planner.computeCenterline(left_boundary, right_boundary, centerline));
  EXPECT_TRUE(centerline.empty());
}
