#pragma once
#include <vector>
#include <utility>

struct BoundaryPoint
{
  double x;
  double y;
};

class TTAPlanner
{
public:
  bool computeCenterline(
    const std::vector<BoundaryPoint> & left_boundary,
    const std::vector<BoundaryPoint> & right_boundary,
    std::vector<BoundaryPoint> & centerline
  );

private:
  std::vector<BoundaryPoint> preprocessBoundaries(
    const std::vector<BoundaryPoint> & boundary
  );

  std::vector<BoundaryPoint> orderBoundary(
    const std::vector<BoundaryPoint> & boundary
  );

  std::vector<std::pair<BoundaryPoint, BoundaryPoint>> pairBoundaryPoints(
    const std::vector<BoundaryPoint> & left_chain,
    const std::vector<BoundaryPoint> & right_chain
  );

  std::vector<BoundaryPoint> computeMidpoints(
    const std::vector<std::pair<BoundaryPoint, BoundaryPoint>> & pairs
  );

  std::vector<BoundaryPoint> orderLoop(
    const std::vector<BoundaryPoint> & centerline
  );

  std::vector<BoundaryPoint> smooth(
    const std::vector<BoundaryPoint> & ordered_centerline
  );

  std::vector<BoundaryPoint> removeLargeJumps(
  const std::vector<BoundaryPoint> & ordered_points);

  std::vector<BoundaryPoint> resample(
  const std::vector<BoundaryPoint> & pts,
  double spacing);

  std::vector<BoundaryPoint> adaptiveResample(
    const std::vector<BoundaryPoint>& pts);
};
