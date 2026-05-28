#include <cmath>
#include <iostream>

#include <Eigen/Geometry>

#include "svo/camera.h"
#include "svo/geometry.h"

static int failures = 0;

static void check(bool ok, const char *name) {
  if (!ok) {
    std::cerr << "FAIL: " << name << "\n";
    ++failures;
  }
}

static bool near(double a, double b, double tol = 1e-9) {
  return std::abs(a - b) < tol;
}

static bool matNear(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                    double tol = 1e-9) {
  return (A - B).norm() < tol;
}

// ---------------------------------------------------------------------------
// Camera::triangulateRectified
// ---------------------------------------------------------------------------
static void testTriangulate() {
  svo::Camera cam;
  cam.fx = 718.856;
  cam.fy = 718.856;
  cam.cx = 607.193;
  cam.cy = 185.216;
  cam.baseline = 0.537150;

  // Known disparity → known depth: depth = fx * baseline / disparity
  {
    const double d = 10.0;
    const double ul = 400.0, vl = 200.0, ur = ul - d;
    Eigen::Vector3d p;
    check(cam.triangulateRectified(ul, vl, ur, p), "tri: valid disparity returns true");
    const double expected_depth = cam.fx * cam.baseline / d;
    check(near(p.z(), expected_depth), "tri: depth from known disparity");
    check(near(p.x(), (ul - cam.cx) * p.z() / cam.fx), "tri: x coordinate");
    check(near(p.y(), (vl - cam.cy) * p.z() / cam.fy), "tri: y coordinate");
  }

  // Zero disparity (ul == ur) → false
  {
    Eigen::Vector3d p;
    check(!cam.triangulateRectified(400.0, 200.0, 400.0, p),
          "tri: zero disparity returns false");
  }

  // Negative disparity (right x > left x) → false
  {
    Eigen::Vector3d p;
    check(!cam.triangulateRectified(400.0, 200.0, 410.0, p),
          "tri: negative disparity returns false");
  }

  // Depth positive for valid point
  {
    Eigen::Vector3d p;
    cam.triangulateRectified(400.0, 200.0, 390.0, p);
    check(p.z() > 0.0, "tri: depth is positive");
  }
}

// ---------------------------------------------------------------------------
// svo::poseWcFromCw / svo::poseCwFromWc
// ---------------------------------------------------------------------------
static void testPoseConversions() {
  // Identity: T_wc = I → R_cw = I, t_cw = 0
  {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    svo::poseCwFromWc(Eigen::Matrix4d::Identity(), R, t);
    check(matNear(R, Eigen::Matrix3d::Identity()), "pose: identity → R_cw = I");
    check(matNear(t, Eigen::Vector3d::Zero()),     "pose: identity → t_cw = 0");
  }

  // poseWcFromCw with identity → T_wc = I
  {
    const Eigen::Matrix4d T =
        svo::poseWcFromCw(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    check(matNear(T, Eigen::Matrix4d::Identity()), "pose: I,0 → T_wc = I");
  }

  // Pure translation: t_cw = (1,2,3) → t_wc = -(1,2,3)
  {
    const Eigen::Vector3d t_cw(1.0, 2.0, 3.0);
    const Eigen::Matrix4d T =
        svo::poseWcFromCw(Eigen::Matrix3d::Identity(), t_cw);
    check(matNear(T.block<3, 1>(0, 3), -t_cw), "pose: pure translation sign");
  }

  // Round-trip (R,t) → T_wc → (R,t)
  {
    const Eigen::Matrix3d R_cw =
        Eigen::AngleAxisd(0.5, Eigen::Vector3d(1, 1, 0).normalized())
            .toRotationMatrix();
    const Eigen::Vector3d t_cw(1.5, -2.3, 0.7);

    const Eigen::Matrix4d T_wc = svo::poseWcFromCw(R_cw, t_cw);
    Eigen::Matrix3d R_rt;
    Eigen::Vector3d t_rt;
    svo::poseCwFromWc(T_wc, R_rt, t_rt);

    check(matNear(R_rt, R_cw), "pose: round-trip R_cw");
    check(matNear(t_rt, t_cw), "pose: round-trip t_cw");
  }

  // T_wc rotation block is a valid rotation matrix (det = 1, R^T R = I)
  {
    const Eigen::Matrix3d R_cw =
        Eigen::AngleAxisd(1.2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    const Eigen::Matrix4d T_wc =
        svo::poseWcFromCw(R_cw, Eigen::Vector3d(1, 0, 0));
    const Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
    check(near((R_wc.transpose() * R_wc - Eigen::Matrix3d::Identity()).norm(), 0.0,
               1e-9),
          "pose: T_wc rotation block is orthogonal");
    check(near(R_wc.determinant(), 1.0, 1e-9),
          "pose: T_wc rotation block has det = 1");
  }
}

// ---------------------------------------------------------------------------
int main() {
  testTriangulate();
  testPoseConversions();

  if (failures > 0) {
    std::cerr << failures << " test(s) FAILED\n";
    return 1;
  }
  std::cout << "All geometry tests passed\n";
  return 0;
}
