import com.comsol.model.*;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.*;

import java.util.Locale;

/** Fresh, unsaved Cartesian XY particle-tracing microcases. */
public class RunXYMinimalSuite {
  private static final String PHYSICS = "pt";
  private static final String STUDY = "std1";
  private static final String TLIST = "range(0,0.05,0.5)";
  private static final String[] IDS = {
    "ballistic", "electric", "linear_drag", "surface_departure",
    "specular", "stick", "hold"
  };

  private static void require(boolean value, String message) {
    if (!value) throw new IllegalStateException(message);
  }

  private static void emit(String kind, String... values) {
    StringBuilder line = new StringBuilder("XYMIN|").append(kind);
    for (int index = 0; index < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    System.out.println(line);
  }

  private static String outputRoot() {
    return ".";
  }

  private static String baseSolver(Model model) {
    String[] direct = model.study(STUDY).getSolverSequences("SolverSequence");
    if (direct.length > 0) return direct[0];
    for (String tag : model.study(STUDY).getSolverSequences("All")) {
      try {
        if (STUDY.equals(model.sol(tag).study())) return tag;
      } catch (Throwable ignored) {
      }
    }
    throw new IllegalStateException("No solver sequence for " + STUDY);
  }

  private static String resultStore(Model model, String base) {
    for (String kind : new String[] {"ParametricStore", "Parametric"}) {
      String[] values = model.study(STUDY).getSolverSequences(kind);
      if (values.length > 0) return values[0];
    }
    return base;
  }

  private static String wallCondition(String id) {
    if (id.equals("specular")) return "Bounce";
    if (id.equals("stick")) return "Stick";
    if (id.equals("hold")) return "Freeze";
    return "Bounce";
  }

  private static String[] initialPosition(String id) {
    if (id.equals("surface_departure")) return new String[] {"0.4[m]", "0[m]"};
    if (id.equals("specular") || id.equals("stick") || id.equals("hold")) {
      return new String[] {"0.8[m]", "0.4[m]"};
    }
    return new String[] {"0.2[m]", "0.3[m]"};
  }

  private static String[] initialVelocity(String id) {
    if (id.equals("surface_departure")) {
      return new String[] {"0.1[m/s]", "0.2[m/s]", "0[m/s]"};
    }
    if (id.equals("specular") || id.equals("stick") || id.equals("hold")) {
      return new String[] {"0.8[m/s]", "0.1[m/s]", "0[m/s]"};
    }
    if (id.equals("linear_drag")) {
      return new String[] {"0.5[m/s]", "-0.2[m/s]", "0[m/s]"};
    }
    return new String[] {"0.3[m/s]", "0.2[m/s]", "0[m/s]"};
  }

  private static String particleMass(String id) {
    return id.equals("linear_drag") ? "4e-15[kg]" : "1e-12[kg]";
  }

  private static Model buildModel(String id, double stepSeconds) {
    Model model = ModelUtil.create("XYMIN_" + id);
    model.label("Cartesian XY minimal particle comparison: " + id);
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.component("comp1").geom("geom1").lengthUnit("m");
    model.component("comp1").geom("geom1").create("rect1", "Rectangle");
    model.component("comp1").geom("geom1").feature("rect1")
        .set("size", new String[] {"1[m]", "1[m]"});
    model.component("comp1").geom("geom1").run();
    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").autoMeshSize(4);
    model.component("comp1").mesh("mesh1").run();

    model.component("comp1").physics().create(PHYSICS, "MathParticle", "geom1");
    Physics physics = model.component("comp1").physics(PHYSICS);
    physics.prop("Formulation").setIndex("Formulation", "NewtonianFirstOrder", 0);
    physics.prop("StoreParticleStatusData").set("StoreParticleStatusData", true);
    physics.feature("pp1").set("mp", particleMass(id));
    physics.feature("wall1").set("WallCondition", wallCondition(id));

    if (id.equals("electric")) {
      model.param().set("q_e", "e_const");
      model.param().set("E_x", "2.4966036297843054e6[V/m]");
      physics.create("for1", "Force", 2);
      physics.feature("for1").selection().all();
      physics.feature("for1").set(
          "F", new String[] {"q_e*E_x", "0[N]", "0[N]"});
    } else if (id.equals("linear_drag")) {
      model.param().set("beta", "2[1/s]");
      model.param().set("u_x", "0.1[m/s]");
      model.param().set("u_y", "0.05[m/s]");
      physics.create("for1", "Force", 2);
      physics.feature("for1").selection().all();
      physics.feature("for1").set(
          "F",
          new String[] {
            "4e-15[kg]*beta*(u_x-pt.vx)",
            "4e-15[kg]*beta*(u_y-pt.vy)",
            "0[N]"
          });
    }

    physics.create("relg1", "ReleaseGrid", -1);
    PhysicsFeature release = physics.feature("relg1");
    release.set("GridType", "AllCombinations");
    release.set("x0", initialPosition(id));
    release.set("v0", initialVelocity(id));
    release.set("Nvel", 1);
    release.set("Nvalt", 1);
    model.study().create(STUDY);
    model.study(STUDY).create("time", "Transient");
    model.study(STUDY).feature("time").set("tlist", TLIST);
    model.study(STUDY).feature("time").set("usertol", true);
    model.study(STUDY).feature("time").set("rtol", "1e-10");
    model.study(STUDY).createAutoSequences("all");
    String base = baseSolver(model);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", String.format(Locale.ROOT, "%.17g", stepSeconds));
    transientSolver.set("rtol", "1e-10");
    emit(
        "configuration",
        "scenario", id,
        "step_s", String.format(Locale.ROOT, "%.17g", stepSeconds),
        "coordinate_system", "cartesian_xy",
        "formulation", "NewtonianFirstOrder",
        "force", id.equals("electric") ? "constant_qE" :
            (id.equals("linear_drag") ? "linear_relaxation" : "none"),
        "wall", wallCondition(id),
        "release", id.equals("surface_departure") ? "exact_surface_origin" : "interior",
        "model_saved", "false");
    return model;
  }

  private static void createParticleDataset(Model model, String solution) {
    model.result().dataset().create("particles", "Particle");
    DatasetFeature data = model.result().dataset("particles");
    data.set("solution", solution);
    data.set("posdof", new String[] {"comp1.qx", "comp1.qy"});
    data.set("geom", "geom1");
    data.set("pgeom", "pgeom_pt");
    data.set("pgeomspec", "fromphysics");
    data.set("physicsinterface", PHYSICS);
  }

  private static void exportHistory(Model model, String id, double stepSeconds) {
    String directory = outputRoot() + "/" + id + "/dt_" +
        String.format(Locale.ROOT, "%.3e", stepSeconds).replace('+', '_');
    model.result().export().create("state", "Data");
    ExportFeature export = model.result().export("state");
    export.set("data", "particles");
    export.set(
        "expr",
        new String[] {
          "pt.pidx", "t", "qx", "qy", "pt.vx", "pt.vy", "particlestatus", "pt.fs"
        });
    export.set(
        "unit",
        new String[] {"1", "s", "m", "m", "m/s", "m/s", "1", "1"});
    export.set(
        "descr",
        new String[] {
          "particle_id", "time_s", "x_m", "y_m", "vx_m_per_s", "vy_m_per_s",
          "current_status_code", "final_status_code"
        });
    export.set("filename", directory + "/state_raw_wide.csv");
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    export.set("innerinput", "all");
    export.run();
  }

  private static void runOne(String id, double stepSeconds) throws Exception {
    Model model = null;
    try {
      model = buildModel(id, stepSeconds);
      long started = System.nanoTime();
      model.study(STUDY).run();
      String base = baseSolver(model);
      String solution = resultStore(model, base);
      createParticleDataset(model, solution);
      double[] times = model.sol(solution).getPVals();
      require(times.length == 11, "Expected 11 output times, got " + times.length);
      exportHistory(model, id, stepSeconds);
      emit(
          "solve_pass",
          "scenario", id,
          "step_s", String.format(Locale.ROOT, "%.17g", stepSeconds),
          "seconds", String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
          "output_times", Integer.toString(times.length));
    } catch (Throwable error) {
      error.printStackTrace(System.out);
      emit(
          "run_error",
          "scenario", id,
          "step_s", String.format(Locale.ROOT, "%.17g", stepSeconds),
          "exception", error.getClass().getName(),
          "message", String.valueOf(error.getMessage()));
      throw new RuntimeException(error);
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  public static void main(String[] args) throws Exception {
    ModelUtil.showProgress(false);
    double[] steps = {0.02, 0.01, 0.005};
    emit("run_start", "scenario_count", Integer.toString(IDS.length), "step_count", "3");
    for (String id : IDS) {
      for (double step : steps) runOne(id, step);
    }
    emit("run_pass", "scenario_count", Integer.toString(IDS.length), "model_saved", "false");
  }
}
