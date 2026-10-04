import com.comsol.model.DatasetFeature;
import com.comsol.model.ExportFeature;
import com.comsol.model.Model;
import com.comsol.model.NumericalFeature;
import com.comsol.model.SolverFeature;
import com.comsol.model.StudyFeature;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;
import java.util.Locale;

/**
 * Exports the M3-C1 Case-A thermophoretic PPR closure primitives.
 *
 * <p>This is an external V&V-only, finest-step rerun. The audited MPH is loaded
 * from a disposable copy, is never saved, and the established M3-C0b v6
 * evidence is not read or modified. Bare derivatives are retained beside the
 * expression-level PPR derivatives so the provenance correction is explicit.
 */
public final class RunM3C1CaseA100ThermophoresisPpr {
  private static final String SOURCE = "source_copy.mph";
  private static final String PHYSICS = "fptas";
  private static final String FEATURE = "thpf1";
  private static final String BACKGROUND_STUDY = "stdASf";
  private static final String BACKGROUND_SOLUTION = "sol26";
  private static final String BACKGROUND_DATASET = "dset_AS_field";
  private static final String STEP_DIRECTORY = "dt_0p15625us";
  private static final String STEP = "0.15625[us]";
  private static final String STEP_S = "1.5625e-7";
  private static final String TLIST = "range(0[s],1e-5[s],4.5e-4[s])";

  private static final String PPR_GRADIENT_R = "ppr(d(root.comp1.AS_Tg,r))";
  private static final String PPR_GRADIENT_Z = "ppr(d(root.comp1.AS_Tg,z))";
  private static final String PPR_HEAT_FLUX_R = "-k_mix*ppr(d(root.comp1.AS_Tg,r))";
  private static final String PPR_HEAT_FLUX_Z = "-k_mix*ppr(d(root.comp1.AS_Tg,z))";

  private static final String[] PARTICLE_COLUMNS = {
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "current_status_code",
    "thermophoretic_force_r_N",
    "thermophoretic_force_z_N",
    "gas_temperature_K",
    "gas_thermal_conductivity_W_per_mK",
    "unrecovered_temperature_gradient_r_K_per_m",
    "unrecovered_temperature_gradient_z_K_per_m",
    "ppr_temperature_gradient_r_K_per_m",
    "ppr_temperature_gradient_z_K_per_m",
    "ppr_heat_flux_r_W_per_m2",
    "ppr_heat_flux_z_W_per_m2",
    "particle_diameter_m",
    "background_gas_molar_mass_kg_per_mol"
  };

  private static final String[] PARTICLE_UNITS = {
    "1", "s", "m", "m", "1", "N", "N", "K", "W/(m*K)", "K/m", "K/m",
    "K/m", "K/m", "W/m^2", "W/m^2", "m", "kg/mol"
  };

  private static final String[] MESH_COLUMNS = {
    "r_m",
    "z_m",
    "domain_id",
    "gas_temperature_K",
    "gas_thermal_conductivity_W_per_mK",
    "unrecovered_temperature_gradient_r_K_per_m",
    "unrecovered_temperature_gradient_z_K_per_m",
    "ppr_temperature_gradient_r_K_per_m",
    "ppr_temperature_gradient_z_K_per_m",
    "ppr_heat_flux_r_W_per_m2",
    "ppr_heat_flux_z_W_per_m2"
  };

  private static final String[] MESH_UNITS = {
    "m", "m", "1", "K", "W/(m*K)", "K/m", "K/m", "K/m", "K/m", "W/m^2",
    "W/m^2"
  };

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3C1PPR|").append(type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static String featureValue(PhysicsFeature feature, String property) {
    try {
      String[] values = feature.getStringArray(property);
      if (values != null && values.length == 1) return values[0];
    } catch (Throwable ignored) {
    }
    return feature.getString(property);
  }

  private static void requireFeatureAuthority(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    require(has(physics.feature().tags(), FEATURE), "Missing thermophoretic feature " + FEATURE);
    PhysicsFeature feature = physics.feature(FEATURE);
    require(feature.isActive(), "Thermophoretic feature is inactive");
    require(
        Arrays.equals(feature.selection().entities(), new int[] {3}),
        "Thermophoretic feature must select only domain 3");
    require(
        "Waldmann".equals(featureValue(feature, "ThermophoreticForceModel")),
        "Thermophoretic model is not Waldmann");
    String usePpr = featureValue(feature, "UsePPR");
    require(
        "1".equals(usePpr) || "true".equalsIgnoreCase(usePpr),
        "Thermophoretic feature does not have UsePPR enabled");
    require(
        "root.comp1.AS_Tg".equals(featureValue(feature, "minput_temperature")),
        "Unexpected thermophoretic temperature input");
    require("k_mix".equals(featureValue(feature, "k")), "Unexpected thermal conductivity input");
    require("Mmix".equals(featureValue(feature, "mg")), "Unexpected background molar mass input");
    emit(
        "feature_authority",
        "physics",
        PHYSICS,
        "feature",
        FEATURE,
        "domain",
        "3",
        "model",
        "Waldmann",
        "UsePPR",
        "true",
        "temperature",
        "root.comp1.AS_Tg",
        "conductivity",
        "k_mix",
        "molar_mass",
        "Mmix");
  }

  private static void configurePhysics(Model model, String study) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    for (String tag : new String[] {
        "bf1", "lf1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1"
    }) {
      require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);
    }
    physics.feature("bf1").active(false);
    physics.feature("lf1").active(false);
    for (String tag : new String[] {
        "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1"
    }) {
      physics.feature(tag).active(true);
    }
    require(!physics.feature("bf1").isActive(), "Brownian feature remained active");
    require(!physics.feature("lf1").isActive(), "Saffman feature remained active");
    for (String tag : new String[] {
        "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1",
        "relg1", "wall1", "outin", "outpump", "axi1"
    }) {
      require(physics.feature(tag).isActive(), "Required feature is inactive: " + tag);
    }
    require(
        physics.prop("StoreParticleStatusData").getBoolean("StoreParticleStatusData"),
        "Particle status storage must remain enabled");
    require(
        !physics.prop("StoreExtra").getBoolean("StoreExtra"),
        "Run must retain StoreExtra=false");
    require(
        "1".equals(physics.prop("WallAccuracyOrder").getString("WallAccuracyOrder")),
        "Run must retain WallAccuracyOrder=1");
    physics.feature("auxq").set("StudyStep", study + "/time");
    model.param().set("d0", "100[nm]");
    requireFeatureAuthority(model);
  }

  private static void allOff(StudyFeature step, Model model) {
    for (String tag : model.component("comp1").physics().tags()) {
      try {
        step.setSolveFor("/physics/" + tag, false);
      } catch (Throwable ignored) {
      }
    }
    for (String tag : model.component("comp1").multiphysics().tags()) {
      try {
        step.setSolveFor("/multiphysics/" + tag, false);
      } catch (Throwable ignored) {
      }
    }
  }

  private static String baseSolver(Model model, String study) {
    String[] direct = model.study(study).getSolverSequences("SolverSequence");
    if (direct.length > 0) return direct[0];
    for (String tag : model.study(study).getSolverSequences("All")) {
      try {
        if (study.equals(model.sol(tag).study())) return tag;
      } catch (Throwable ignored) {
      }
    }
    throw new IllegalStateException("No solver sequence for " + study);
  }

  private static String resultStore(Model model, String study, String base) {
    for (String kind : new String[] {"ParametricStore", "Parametric"}) {
      String[] values = model.study(study).getSolverSequences(kind);
      if (values.length > 0) return values[0];
    }
    return base;
  }

  private static String createAndRunStudy(Model model) {
    String study = "stdM3C1Ppr";
    require(!has(model.study().tags(), study), "Study tag already exists: " + study);
    model.param().set("M3C1_ppr_dt", STEP);
    model.study().create(study);
    model.study(study).label("M3-C1 thermophoresis PPR closure, Case A 100 nm");

    model.study(study).create("param", "Parametric");
    StudyFeature parameter = model.study(study).feature("param");
    parameter.set("pname", new String[] {"d0"});
    parameter.set("plistarr", new String[] {"100"});
    parameter.set("punit", new String[] {"nm"});
    parameter.set("sweeptype", "sparse");
    try {
      parameter.set("keepsol", "all");
    } catch (Throwable ignored) {
    }

    model.study(study).create("time", "Transient");
    StudyFeature time = model.study(study).feature("time");
    allOff(time, model);
    time.setSolveFor("/physics/" + PHYSICS, true);
    time.set("tlist", TLIST);
    time.set("usertol", true);
    time.set("rtol", "1e-8");
    time.set("usesol", true);
    time.set("notsolmethod", "sol");
    time.set("notstudy", BACKGROUND_STUDY);
    time.set("notstudystep", "stat");
    time.set("notsol", BACKGROUND_SOLUTION);
    time.set("notsoluse", "current");
    time.set("notsolnum", "last");
    time.label("Particle-only saved-background rerun; fixed classical RK4 " + STEP);

    configurePhysics(model, study);
    model.study(study).createAutoSequences("all");
    String base = baseSolver(model, study);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", "M3C1_ppr_dt");
    transientSolver.set("rtol", "1e-8");

    long started = System.nanoTime();
    model.study(study).run();
    emit(
        "solve_pass",
        "step_s",
        STEP_S,
        "seconds",
        String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
        "base",
        base);
    return resultStore(model, study, base);
  }

  private static void createParticleDataset(Model model, String dataset, String solution) {
    require(!has(model.result().dataset().tags(), dataset), "Dataset tag already exists: " + dataset);
    model.result().dataset().create(dataset, "Particle");
    DatasetFeature data = model.result().dataset(dataset);
    data.set("solution", solution);
    data.set("posdof", new String[] {"comp1.q3r", "comp1.q3z"});
    data.set("geom", "geom1");
    data.set("pgeom", "pgeom_fptas");
    data.set("pgeomspec", "fromphysics");
    data.set("physicsinterface", PHYSICS);
  }

  private static String[] particleExpressions() {
    String[] values = {
      "fptas.pidx",
      "t",
      "q3r",
      "q3z",
      "particlestatus",
      "fptas.thpf1.Ftfr",
      "fptas.thpf1.Ftfz",
      "root.comp1.AS_Tg",
      "k_mix",
      "d(root.comp1.AS_Tg,r)",
      "d(root.comp1.AS_Tg,z)",
      PPR_GRADIENT_R,
      PPR_GRADIENT_Z,
      PPR_HEAT_FLUX_R,
      PPR_HEAT_FLUX_Z,
      "d0",
      "Mmix"
    };
    require(values.length == PARTICLE_COLUMNS.length, "Particle expression/column count mismatch");
    require(PARTICLE_UNITS.length == PARTICLE_COLUMNS.length, "Particle unit/column count mismatch");
    return values;
  }

  private static String[] meshExpressions() {
    String[] values = {
      "r",
      "z",
      "dom",
      "root.comp1.AS_Tg",
      "k_mix",
      "d(root.comp1.AS_Tg,r)",
      "d(root.comp1.AS_Tg,z)",
      PPR_GRADIENT_R,
      PPR_GRADIENT_Z,
      PPR_HEAT_FLUX_R,
      PPR_HEAT_FLUX_Z
    };
    require(values.length == MESH_COLUMNS.length, "Mesh expression/column count mismatch");
    require(MESH_UNITS.length == MESH_COLUMNS.length, "Mesh unit/column count mismatch");
    return values;
  }

  private static ExportFeature createExport(
      Model model,
      String tag,
      String dataset,
      String filename,
      String[] expressions,
      String[] units,
      String[] columns) {
    require(!has(model.result().export().tags(), tag), "Export tag already exists: " + tag);
    model.result().export().create(tag, "Data");
    ExportFeature export = model.result().export(tag);
    export.set("data", dataset);
    export.set("expr", expressions);
    export.set("unit", units);
    export.set("descr", columns);
    export.set("filename", STEP_DIRECTORY + "/" + filename);
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    return export;
  }

  private static void exportParticleStates(Model model, String dataset) {
    String filename = "thermophoresis_ppr_particle_raw_wide.csv";
    ExportFeature export =
        createExport(
            model,
            "m3c1PprParticle",
            dataset,
            filename,
            particleExpressions(),
            PARTICLE_UNITS,
            PARTICLE_COLUMNS);
    export.set("innerinput", "all");
    try {
      export.set("outerinput", "all");
    } catch (Throwable ignored) {
    }
    emit("export_start", "table", filename, "recovery", "explicit_ppr_expression");
    export.run();
    emit("export_pass", "table", filename);
  }

  private static void exportNativeMeshNodes(Model model) {
    require(
        has(model.result().dataset().tags(), BACKGROUND_DATASET),
        "Missing background dataset " + BACKGROUND_DATASET);
    String filename = "thermophoresis_ppr_native_mesh_nodes.csv";
    ExportFeature export =
        createExport(
            model,
            "m3c1PprMesh",
            BACKGROUND_DATASET,
            filename,
            meshExpressions(),
            MESH_UNITS,
            MESH_COLUMNS);
    export.set("outerinput", "last");
    export.set("innerinput", "last");
    export.set("location", "fromdataset");
    export.set("level", "surface");
    export.set("resolution", "normal");
    export.set("smooth", "material");
    export.set("recover", "off");
    emit(
        "export_start",
        "table",
        filename,
        "dataset",
        BACKGROUND_DATASET,
        "location",
        "fromdataset",
        "recover",
        "off",
        "operator",
        "ppr");
    export.run();
    emit("export_pass", "table", filename);
  }

  private static int particleRows(Model model, String dataset) {
    String tag = "m3c1PprParticleCount";
    try {
      model.result().numerical().create(tag, "Particle");
      NumericalFeature numerical = model.result().numerical(tag);
      numerical.set("data", dataset);
      numerical.set("expr", "fptas.pidx");
      numerical.set("unit", "1");
      numerical.set("innerinput", "first");
      int count = 0;
      for (double[] row : numerical.getReal(false)) count += row.length;
      return count;
    } finally {
      try {
        model.result().numerical().remove(tag);
      } catch (Throwable ignored) {
      }
    }
  }

  private static void run() throws Exception {
    Model model = null;
    try {
      model = ModelUtil.loadCopy("M3C1ThermophoresisPpr", SOURCE);
      String solution = createAndRunStudy(model);
      String particleDataset = "partM3C1Ppr";
      createParticleDataset(model, particleDataset, solution);
      double[] times = model.sol(solution).getPVals();
      require(times.length == 46, "Expected 46 output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(Math.abs(times[times.length - 1] - 4.5e-4) < 1e-14, "Unexpected final time");
      int particles = particleRows(model, particleDataset);
      require(particles == 287, "Expected 287 particles, got " + particles);
      exportParticleStates(model, particleDataset);
      exportNativeMeshNodes(model);
      emit(
          "configuration",
          "step_s",
          STEP_S,
          "brownian_active",
          "false",
          "saffman_active",
          "false",
          "dynamic_charge_active",
          "true",
          "store_particle_status",
          "true",
          "store_extra",
          "false",
          "wall_accuracy_order",
          "1",
          "integrator",
          "classical_rk4",
          "integrator_order",
          "4",
          "relative_tolerance",
          "1e-8",
          "background_study",
          BACKGROUND_STUDY,
          "background_solution",
          BACKGROUND_SOLUTION,
          "physics",
          PHYSICS,
          "feature",
          FEATURE,
          "UsePPR",
          "true",
          "ppr_gradient_r",
          PPR_GRADIENT_R,
          "ppr_gradient_z",
          PPR_GRADIENT_Z,
          "ppr_heat_flux_r",
          PPR_HEAT_FLUX_R,
          "ppr_heat_flux_z",
          PPR_HEAT_FLUX_Z,
          "output_times",
          Integer.toString(times.length),
          "particle_rows",
          Integer.toString(particles),
          "source_model",
          SOURCE,
          "model_saved",
          "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  public static void main(String[] args) throws Exception {
    ModelUtil.showProgress(false);
    run();
    emit(
        "run_pass",
        "case",
        "caseA_100nm",
        "step_s",
        STEP_S,
        "steps_run",
        "1",
        "particle_table",
        "thermophoresis_ppr_particle_raw_wide.csv",
        "mesh_table",
        "thermophoresis_ppr_native_mesh_nodes.csv",
        "model_saved",
        "false");
  }
}
