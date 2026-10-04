import com.comsol.model.DatasetFeature;
import com.comsol.model.ExportFeature;
import com.comsol.model.FunctionFeature;
import com.comsol.model.Model;
import com.comsol.model.NumericalFeature;
import com.comsol.model.SolverFeature;
import com.comsol.model.StudyFeature;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.ModelUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Runs a no-save M3-C2 100 nm COMSOL stochastic campaign.
 *
 * <p>The source model contributes geometry, boundary selections, particle
 * topology, and the saved stationary background solution. All particle-RHS
 * primitives and the realized release state are rebound to the same canonical
 * P1 package used by the candidate solver. Brownian forcing is the built-in
 * COMSOL feature, is restricted to the saved R-Z degrees of freedom, uses the
 * Epstein-equivalent viscosity, and takes its only effective replica argument
 * from the contract-selected {@code bf1.i} parameter. Each request is loaded
 * from a fresh copy and the model is never saved.
 */
public final class RunM3C2StochasticCampaign {
  private static final String SOURCE = "source_copy.mph";
  private static final String PHYSICS = RunM3C2StochasticRequest.physicsTag();
  private static final String BACKGROUND_STUDY = RunM3C2StochasticRequest.backgroundStudy();
  private static final String BACKGROUND_SOLUTION = RunM3C2StochasticRequest.backgroundSolution();
  private static final String TLIST =
      "range(0[s],1e-5[s],5e-4[s]) "
          + "range(6e-4[s],1e-4[s],5e-3[s]) "
          + "range(6e-3[s],1e-3[s],3e-2[s])";
  private static final int EXPECTED_OUTPUT_TIMES = 121;
  private static final int EXPECTED_PARTICLES = 287;
  private static final long TIME_END_NS = 30000000L;
  private static final int MAX_REQUESTS = 32;

  private static final String[] P1_NAMES = {
    "m3c1_rhog", "m3c1_mug", "m3c1_Tg", "m3c1_lambdag",
    "m3c1_ne", "m3c1_ni", "m3c1_Te", "m3c1_TiV", "m3c1_mi",
    "m3c1_lambdaD", "m3c1_lambdaIn", "m3c1_omegaPhi",
    "m3c1_ugr", "m3c1_ugz", "m3c1_Er", "m3c1_Ez",
    "m3c1_uir", "m3c1_uiz", "m3c1_gradE2r", "m3c1_gradE2z",
    "m3c1_qr", "m3c1_qz"
  };
  private static final String[] P1_UNITS = {
    "kg/m^3", "Pa*s", "K", "m", "1/m^3", "1/m^3", "V", "V", "kg",
    "m", "m", "1/s", "m/s", "m/s", "V/m", "V/m", "m/s", "m/s",
    "V^2/m^3", "V^2/m^3", "W/m^2", "W/m^2"
  };
  private static final String[] RELEASE_NAMES = {"m3c1_vr0", "m3c1_vz0", "m3c1_Z0"};
  private static final String[] RELEASE_UNITS = {"m/s", "m/s", "1"};

  private static final String THERMO_R =
      "(32/15)*(d0/2)^2*m3c1_qr(r,z)/"
          + "sqrt(8*k_B_const*m3c1_Tg(r,z)/(pi*1.2753471408396638e-25[kg]))";
  private static final String THERMO_Z =
      "(32/15)*(d0/2)^2*m3c1_qz(r,z)/"
          + "sqrt(8*k_B_const*m3c1_Tg(r,z)/(pi*1.2753471408396638e-25[kg]))";
  private static final String LIFT_R =
      "pi*m3c1_rhog(r,z)*m3c1_lambdag(r,z)*(d0/2)^2*"
          + "(m3c1_ugz(r,z)-" + PHYSICS + ".vz)*m3c1_omegaPhi(r,z)";
  private static final String LIFT_Z =
      "-pi*m3c1_rhog(r,z)*m3c1_lambdag(r,z)*(d0/2)^2*"
          + "(m3c1_ugr(r,z)-" + PHYSICS + ".vr)*m3c1_omegaPhi(r,z)";
  private static final String DEP_FACTOR =
      "2*pi*epsilon0_const*1*(d0/2)^3*0.5161290322580645";
  private static final String DEP_R = DEP_FACTOR + "*m3c1_gradE2r(r,z)";
  private static final String DEP_Z = DEP_FACTOR + "*m3c1_gradE2z(r,z)";

  private static final String[] STATE_COLUMNS = {
    "particle_id", "time_s", "r_m", "z_m", "velocity_r_m_per_s",
    "velocity_z_m_per_s", "charge_number_e", "current_status_code",
    "final_status_code", "stop_or_event_time_s"
  };
  private static final String[] STATE_UNITS = {
    "1", "s", "m", "m", "m/s", "m/s", "1", "1", "1", "s"
  };

  private RunM3C2StochasticCampaign() {}

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3C2_COMSOL|").append(type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static String stepLabel(long stepNs) {
    if (stepNs % 1000L == 0L) return "dt_" + Long.toString(stepNs / 1000L) + "us";
    return "dt_" + Long.toString(stepNs) + "ns";
  }

  private static List<String[]> requests() {
    List<String[]> requests = new ArrayList<>();
    Set<String> keys = new HashSet<>();
    for (String row : RunM3C2StochasticRequest.rows()) {
      require(!row.isEmpty(), "Blank row in M3-C2 execution request");
      String[] values = row.split(",", -1);
      require(values.length == 3, "Invalid M3-C2 execution-request row");
      int seed = Integer.parseInt(values[0]);
      long stepNs = Long.parseLong(values[1]);
      require(seed >= 0, "M3-C2 seed must be nonnegative");
      require(stepNs > 0L && TIME_END_NS % stepNs == 0L,
          "M3-C2 step must be positive and divide 30 ms");
      String directory =
          "levels/" + stepLabel(stepNs) + "/replicas/seed_" + Integer.toString(seed);
      require(directory.equals(values[2]), "Unsafe or inconsistent M3-C2 output directory");
      String key = Integer.toString(seed) + ":" + Long.toString(stepNs);
      require(keys.add(key), "Duplicate M3-C2 execution request: " + key);
      requests.add(new String[] {values[0], values[1], values[2]});
    }
    require(!requests.isEmpty(), "The M3-C2 pilot request is empty");
    require(requests.size() <= MAX_REQUESTS, "Too many M3-C2 execution requests");
    return requests;
  }

  private static int seed(String[] request) {
    return Integer.parseInt(request[0]);
  }

  private static long stepNs(String[] request) {
    return Long.parseLong(request[1]);
  }

  private static String directory(String[] request) {
    return request[2];
  }

  private static String stepExpression(long stepNs) {
    return Long.toString(stepNs) + "[ns]";
  }

  private static String stepSeconds(long stepNs) {
    return String.format(Locale.ROOT, "%.17g", stepNs * 1.0e-9);
  }

  private static void createSectionwiseFunctions(Model model) {
    require(P1_NAMES.length == P1_UNITS.length, "P1 function metadata mismatch");
    for (int index = 0; index < P1_NAMES.length; index++) {
      String tag = "m3c2P1F" + index;
      model.func().create(tag, "Interpolation");
      FunctionFeature function = model.func(tag);
      function.set("source", "file");
      function.set("filename", P1_NAMES[index] + "_sectionwise.txt");
      function.set("struct", "sectionwise");
      function.set("funcs", new String[][] {{P1_NAMES[index], "1"}});
      function.set("interp", "linear");
      function.set("extrap", "const");
      function.set("argunit", new String[] {"m", "m"});
      function.set("fununit", P1_UNITS[index]);
      function.importData();
    }
  }

  private static void createReleaseFunctions(Model model) {
    require(RELEASE_NAMES.length == RELEASE_UNITS.length, "Release metadata mismatch");
    for (int index = 0; index < RELEASE_NAMES.length; index++) {
      String tag = "m3c2ReleaseF" + index;
      model.func().create(tag, "Interpolation");
      FunctionFeature function = model.func(tag);
      function.set("source", "file");
      function.set("filename", RELEASE_NAMES[index] + ".txt");
      function.set("struct", "spreadsheet");
      function.set("funcs", new String[][] {{RELEASE_NAMES[index], "1"}});
      function.set("interp", "linear");
      function.set("extrap", "const");
      function.set("argunit", new String[] {"m", "m"});
      function.set("fununit", RELEASE_UNITS[index]);
      function.importData();
    }
  }

  private static void bindP1Variables(Model model) {
    String[][] caseABindings = {
      {"AS_rhog", "m3c1_rhog(r,z)"}, {"AS_mug", "m3c1_mug(r,z)"},
      {"AS_Tg", "m3c1_Tg(r,z)"}, {"AS_lambdag", "m3c1_lambdag(r,z)"},
      {"AS_ne", "m3c1_ne(r,z)"}, {"AS_ni", "m3c1_ni(r,z)"},
      {"AS_TiV", "m3c1_TiV(r,z)"}, {"AS_lambdaD", "m3c1_lambdaD(r,z)"},
      {"AS_lambda_in", "m3c1_lambdaIn(r,z)"}, {"AS_ugr", "m3c1_ugr(r,z)"},
      {"AS_ugz", "m3c1_ugz(r,z)"}, {"AS_Er", "m3c1_Er(r,z)"},
      {"AS_Ez", "m3c1_Ez(r,z)"}, {"AS_uir", "m3c1_uir(r,z)"},
      {"AS_uiz", "m3c1_uiz(r,z)"},
      {"AS_Ge0", "pi*(d0/2)^2*m3c1_ne(r,z)*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const))"},
      {"AS_phi1", "e_const/(4*pi*epsilon0_const*(d0/2)*(1+(d0/2)/m3c1_lambdaD(r,z)))"},
      {"AS_pabs", RunM3C2StochasticRequest.pressureExpression()}
    };
    String[][] casePBindings = {
      {"rho_g_d", "m3c1_rhog(r,z)"}, {"mu_g_d", "m3c1_mug(r,z)"},
      {"lambda_g_d", "m3c1_lambdag(r,z)"}, {"ne_d", "m3c1_ne(r,z)"},
      {"ni_d", "m3c1_ni(r,z)"}, {"Te_d", "m3c1_Te(r,z)"},
      {"TiV_d", "m3c1_TiV(r,z)"}, {"mi_d", "m3c1_mi(r,z)"},
      {"lambdaD_d", "m3c1_lambdaD(r,z)"}, {"lambda_in_d", "m3c1_lambdaIn(r,z)"},
      {"uir_d", "m3c1_uir(r,z)"}, {"uiz_d", "m3c1_uiz(r,z)"},
      {"Ge0_d", "pi*(d0/2)^2*m3c1_ne(r,z)*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const))"},
      {"phi1_d", "e_const/(4*pi*epsilon0_const*(d0/2)*(1+(d0/2)/m3c1_lambdaD(r,z)))"},
      {"muB_d", "m3c1_mug(r,z)/(36*(m3c1_lambdag(r,z)/d0)/(8+pi*sigmaR_p))"}
    };
    String[][] bindings = RunM3C2StochasticRequest.isCaseP() ? casePBindings : caseABindings;
    String variableTag = RunM3C2StochasticRequest.sharedVariableTag();
    require(has(model.component("comp1").variable().tags(), variableTag),
        "Missing shared variable group " + variableTag);
    for (String[] binding : bindings) {
      model.component("comp1").variable(variableTag).set(binding[0], binding[1]);
    }
  }

  private static String p1ProducerExpression(String source) {
    String[][] caseAReplacements = {
      {"root.comp1.AS_lambda_in", "m3c1_lambdaIn(r,z)"},
      {"root.comp1.AS_lambdaD", "m3c1_lambdaD(r,z)"},
      {"root.comp1.AS_uir", "m3c1_uir(r,z)"},
      {"root.comp1.AS_uiz", "m3c1_uiz(r,z)"},
      {"root.comp1.AS_TiV", "m3c1_TiV(r,z)"},
      {"root.comp1.AS_ni", "m3c1_ni(r,z)"},
      {"root.comp1.AS_Ge0", "(pi*(d0/2)^2*m3c1_ne(r,z)*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const)))"},
      {"root.comp1.AS_phi1", "(e_const/(4*pi*epsilon0_const*(d0/2)*(1+(d0/2)/m3c1_lambdaD(r,z))))"},
      {"root.comp1.AS_mi", "m3c1_mi(r,z)"},
      {"root.comp1.AS_Te", "m3c1_Te(r,z)"},
      {"AS_mi", "m3c1_mi(r,z)"}, {"AS_Te", "m3c1_Te(r,z)"}
    };
    String[][] casePReplacements = {
      {"root.comp1.lambda_in_d", "m3c1_lambdaIn(r,z)"},
      {"root.comp1.lambdaD_d", "m3c1_lambdaD(r,z)"},
      {"root.comp1.uir_d", "m3c1_uir(r,z)"},
      {"root.comp1.uiz_d", "m3c1_uiz(r,z)"},
      {"root.comp1.TiV_d", "m3c1_TiV(r,z)"},
      {"root.comp1.ni_d", "m3c1_ni(r,z)"},
      {"root.comp1.Ge0_d", "(pi*(d0/2)^2*m3c1_ne(r,z)*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const)))"},
      {"root.comp1.phi1_d", "(e_const/(4*pi*epsilon0_const*(d0/2)*(1+(d0/2)/m3c1_lambdaD(r,z))))"},
      {"root.comp1.mi_d", "m3c1_mi(r,z)"},
      {"root.comp1.Te_d", "m3c1_Te(r,z)"}
    };
    String[][] replacements =
        RunM3C2StochasticRequest.isCaseP() ? casePReplacements : caseAReplacements;
    String result = source;
    for (String[] replacement : replacements) result = result.replace(replacement[0], replacement[1]);
    String[] caseAForbidden = {
        "root.comp1.AS_lambda_in", "root.comp1.AS_lambdaD", "root.comp1.AS_uir",
        "root.comp1.AS_uiz", "root.comp1.AS_TiV", "root.comp1.AS_ni",
        "root.comp1.AS_Ge0", "root.comp1.AS_phi1", "root.comp1.AS_mi", "root.comp1.AS_Te"
    };
    String[] casePForbidden = {
        "root.comp1.lambda_in_d", "root.comp1.lambdaD_d", "root.comp1.uir_d",
        "root.comp1.uiz_d", "root.comp1.TiV_d", "root.comp1.ni_d",
        "root.comp1.Ge0_d", "root.comp1.phi1_d", "root.comp1.mi_d", "root.comp1.Te_d"
    };
    String[] forbiddenFields =
        RunM3C2StochasticRequest.isCaseP() ? casePForbidden : caseAForbidden;
    for (String forbidden : forbiddenFields) {
      require(!result.contains(forbidden), "Native producer field remained: " + forbidden);
    }
    return result;
  }

  private static void bindProducerFormulas(Physics physics) {
    String[] rate = physics.feature("auxq").getStringArray("R");
    require(rate.length >= 1, "Missing dynamic-charge formula");
    physics.feature("auxq").set("R", new String[] {p1ProducerExpression(rate[0])});
    String[] ion = physics.feature("idf").getStringArray("F");
    require(ion.length >= 3, "Missing ion-drag formula");
    physics.feature("idf").set("F", new String[] {
        p1ProducerExpression(ion[0]), p1ProducerExpression(ion[1]), p1ProducerExpression(ion[2])});
  }

  private static void configureCustomForce(
      PhysicsFeature feature, String label, String radial, String axial, String study) {
    feature.label(label);
    feature.selection().set(3);
    feature.set("SpecifyForce", "Directly");
    feature.set("F", new String[] {radial, "0[N]", axial});
    feature.set("ParticlesToAffect", "All");
    feature.set("AffectedParticleProperties", "pp1");
    feature.set("StudyStep", study + "/time");
  }

  private static void requireFeatures(Physics physics) {
    for (String tag : new String[] {
        "bf1", "lf1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm",
        "depf", "gf1", "pp1", "relg1", "wall1", "outin", "outpump", "axi1"
    }) require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);
  }

  private static void configurePhysics(Model model, String study, int seed) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    requireFeatures(physics);
    require("0".equals(physics.prop("IncludeOutOfPlane").getString("IncludeOutOfPlane")),
        "The particle interface must remain R-Z with out-of-plane disabled");

    physics.prop("RandomNumberArgs").set("RandomNumberArgs", "UserDefined");
    physics.feature("bf1").active(true);
    physics.feature("lf1").active(false);
    physics.feature("thpf1").active(false);
    for (String tag : new String[] {"auxq", "idf", "ef1", "df1", "liftfm", "depf", "gf1"}) {
      physics.feature(tag).active(true);
    }

    bindP1Variables(model);
    bindProducerFormulas(physics);
    model.param().set("d0", "100[nm]");
    model.param().set("sigmaR_p", "0.9");
    model.param().set(RunM3C2StochasticRequest.seedParameter(), Integer.toString(seed));

    physics.feature("relg1").set("v0", new String[] {"m3c1_vr0(r,z)", "0[m/s]", "m3c1_vz0(r,z)"});
    physics.feature("relg1").set("aux0_auxq", "m3c1_Z0(r,z)");
    physics.feature("pp1").set("ChargeSpecification", "UserDefined");
    physics.feature("pp1").set("Z", RunM3C2StochasticRequest.chargeStateExpression());

    physics.feature("ef1").set("E", new String[] {"m3c1_Er(r,z)", "0[V/m]", "m3c1_Ez(r,z)"});
    physics.feature("df1").set("u", new String[] {"m3c1_ugr(r,z)", "0[m/s]", "m3c1_ugz(r,z)"});
    physics.feature("df1").set("rho", "m3c1_rhog(r,z)");
    physics.feature("df1").set("mu", "m3c1_mug(r,z)");
    physics.feature("df1").set("minput_temperature", "m3c1_Tg(r,z)");
    physics.feature("df1").set("pA", RunM3C2StochasticRequest.pressureExpression());
    physics.feature("df1").set("minput_pressure", RunM3C2StochasticRequest.pressureExpression());
    physics.feature("df1").set("S", "1.0");
    physics.feature("df1").set("sigmaR", "sigmaR_p");
    physics.feature("gf1").set("rho", "m3c1_rhog(r,z)");
    physics.feature("gf1").set("minput_temperature", "m3c1_Tg(r,z)");

    PhysicsFeature brownian = physics.feature("bf1");
    brownian.set("mu_mat", "userdef");
    brownian.set("mu", RunM3C2StochasticRequest.viscosityExpression());
    brownian.set("i", RunM3C2StochasticRequest.seedParameter());
    brownian.set("minput_temperature_src", "userdef");
    brownian.set("minput_temperature", RunM3C2StochasticRequest.temperatureExpression());
    brownian.set("minput_pressure_src", "userdef");
    brownian.set("minput_pressure", RunM3C2StochasticRequest.pressureExpression());
    brownian.set("StudyStep", study + "/time");

    physics.feature("auxq").set("StudyStep", study + "/time");
    physics.feature("idf").set("StudyStep", study + "/time");
    configureCustomForce(physics.feature("liftfm"), "M3-C2 common-P1 free-molecular lift", LIFT_R, LIFT_Z, study);
    configureCustomForce(physics.feature("depf"), "M3-C2 common-P1 stored-gradient DEP", DEP_R, DEP_Z, study);
    String thermoTag = "m3c2HeatFlux";
    require(!has(physics.feature().tags(), thermoTag), "M3-C2 force tag already exists");
    physics.create(thermoTag, "Force", 2);
    configureCustomForce(physics.feature(thermoTag), "M3-C2 stored-heat-flux Waldmann force", THERMO_R, THERMO_Z, study);

    physics.feature("wall1").set("WallCondition", "Stick");
    physics.feature("outin").set("WallCondition", "Freeze");
    physics.feature("outpump").set("WallCondition", "Disappear");

    require("UserDefined".equals(physics.prop("RandomNumberArgs").getString("RandomNumberArgs")),
        "COMSOL random-number mode is not UserDefined");
    require(RunM3C2StochasticRequest.seedParameter().equals(brownian.getString("i")),
        "bf1.i is not seed authority");
    require(RunM3C2StochasticRequest.viscosityExpression().equals(brownian.getString("mu")),
        "bf1 viscosity changed");
    require(brownian.isActive(), "Brownian force remained inactive");
    require(!physics.feature("lf1").isActive(), "Saffman force must remain inactive");
    for (String tag : new String[] {
        "auxq", "idf", "ef1", "df1", "liftfm", "depf", "gf1", thermoTag,
        "relg1", "wall1", "outin", "outpump", "axi1"
    }) require(physics.feature(tag).isActive(), "Required feature inactive: " + tag);
    require(physics.prop("StoreParticleStatusData").getBoolean("StoreParticleStatusData"),
        "Particle status storage must remain enabled");
    require(!physics.prop("StoreExtra").getBoolean("StoreExtra"), "StoreExtra must remain false");
    require("1".equals(physics.prop("WallAccuracyOrder").getString("WallAccuracyOrder")),
        "Brownian wall accuracy order must be one");
  }

  private static void disableAll(StudyFeature step, Model model) {
    for (String tag : model.component("comp1").physics().tags()) {
      step.setSolveFor("/physics/" + tag, false);
    }
    for (String tag : model.component("comp1").multiphysics().tags()) {
      step.setSolveFor("/multiphysics/" + tag, false);
    }
  }

  private static String solveForMap(StudyFeature step, String kind, String[] sourceTags) {
    String[] tags = sourceTags.clone();
    Arrays.sort(tags);
    if (tags.length == 0) return "none";
    StringBuilder result = new StringBuilder();
    for (String tag : tags) {
      if (result.length() > 0) result.append(',');
      result.append(tag).append('=').append(step.solveFor("/" + kind + "/" + tag));
    }
    return result.toString();
  }

  private static String[] requireParticleOnly(StudyFeature step, Model model) {
    String[] physicsTags = model.component("comp1").physics().tags();
    require(has(physicsTags, PHYSICS), "Source model is missing particle physics " + PHYSICS);
    String physics = solveForMap(step, "physics", physicsTags);
    String multiphysics =
        solveForMap(step, "multiphysics", model.component("comp1").multiphysics().tags());
    for (String tag : physicsTags) {
      require(
          step.solveFor("/physics/" + tag) == PHYSICS.equals(tag),
          "Study solve-for state is not particle-interface-only: " + tag);
    }
    for (String tag : model.component("comp1").multiphysics().tags()) {
      require(
          !step.solveFor("/multiphysics/" + tag),
          "Study retained multiphysics solve-for coupling: " + tag);
    }
    return new String[] {physics, multiphysics};
  }

  private static String baseSolver(Model model, String study) {
    String[] direct = model.study(study).getSolverSequences("SolverSequence");
    if (direct.length > 0) return direct[0];
    for (String tag : model.study(study).getSolverSequences("All")) {
      try { if (study.equals(model.sol(tag).study())) return tag; } catch (Throwable ignored) {}
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

  private static String[] createAndRunStudy(Model model, String[] request) {
    String study = "stdM3C2";
    model.param().set("M3C2_dt", stepExpression(stepNs(request)));
    model.study().create(study);
    model.study(study).label("M3-C2 common-P1 stochastic campaign 100 nm");
    model.study(study).create("time", "Transient");
    StudyFeature time = model.study(study).feature("time");
    disableAll(time, model);
    time.setSolveFor("/physics/" + PHYSICS, true);
    requireParticleOnly(time, model);
    time.set("tlist", TLIST);
    time.set("usertol", true);
    time.set("rtol", "1e-2");
    time.set("usesol", true);
    time.set("notsolmethod", "sol");
    time.set("notstudy", BACKGROUND_STUDY);
    time.set("notstudystep", RunM3C2StochasticRequest.backgroundStudyStep());
    time.set("notsol", BACKGROUND_SOLUTION);
    time.set("notsoluse", "current");
    time.set("notsolnum", "last");

    configurePhysics(model, study, seed(request));
    model.study(study).createAutoSequences("all");
    String[] solveFor = requireParticleOnly(time, model);
    String base = baseSolver(model, study);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", "M3C2_dt");
    transientSolver.set("rtol", "1e-2");

    long started = System.nanoTime();
    model.study(study).run();
    emit("solve_pass", "seed", Integer.toString(seed(request)), "step_s", stepSeconds(stepNs(request)),
        "seconds", String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
        "base", base, "directory", directory(request));
    return new String[] {resultStore(model, study, base), solveFor[0], solveFor[1]};
  }

  private static void createParticleDataset(Model model, String solution) {
    String dataset = "partM3C2";
    model.result().dataset().create(dataset, "Particle");
    DatasetFeature data = model.result().dataset(dataset);
    data.set("solution", solution);
    data.set("posdof", new String[] {
        "comp1." + RunM3C2StochasticRequest.positionRExpression(),
        "comp1." + RunM3C2StochasticRequest.positionZExpression()
    });
    data.set("geom", "geom1");
    data.set("pgeom", RunM3C2StochasticRequest.particleGeometry());
    data.set("pgeomspec", "fromphysics");
    data.set("physicsinterface", PHYSICS);
  }

  private static String[] stateExpressions(Model model) {
    String[] values = {
      PHYSICS + ".pidx", "t", RunM3C2StochasticRequest.positionRExpression(),
      RunM3C2StochasticRequest.positionZExpression(),
      RunM3C2StochasticRequest.velocityRExpression(),
      RunM3C2StochasticRequest.velocityZExpression(),
      RunM3C2StochasticRequest.chargeStateExpression(),
      "particlestatus", PHYSICS + ".fs", PHYSICS + ".st"
    };
    require(values.length == STATE_COLUMNS.length && values.length == STATE_UNITS.length,
        "State export metadata mismatch");
    return values;
  }

  private static void exportHistory(Model model, String[] request) {
    String tag = "m3c2Trajectory";
    model.result().export().create(tag, "Data");
    ExportFeature export = model.result().export(tag);
    export.set("data", "partM3C2");
    export.set("expr", stateExpressions(model));
    export.set("unit", STATE_UNITS);
    export.set("descr", STATE_COLUMNS);
    export.set("filename", directory(request) + "/trajectory_raw_wide.csv");
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    export.set("innerinput", "all");
    export.run();
  }

  private static int particleRows(Model model) {
    String tag = "m3c2ParticleCount";
    try {
      model.result().numerical().create(tag, "Particle");
      NumericalFeature numerical = model.result().numerical(tag);
    numerical.set("data", "partM3C2");
    numerical.set("expr", PHYSICS + ".pidx");
      numerical.set("unit", "1");
      numerical.set("innerinput", "first");
      int count = 0;
      for (double[] row : numerical.getReal(false)) count += row.length;
      return count;
    } finally {
      try { model.result().numerical().remove(tag); } catch (Throwable ignored) {}
    }
  }

  private static void runOne(String[] request) throws Exception {
    Model model = null;
    try {
      model = ModelUtil.loadCopy("M3C2" + seed(request) + "Dt" + stepNs(request), SOURCE);
      createSectionwiseFunctions(model);
      createReleaseFunctions(model);
      String[] studyRun = createAndRunStudy(model, request);
      createParticleDataset(model, studyRun[0]);
      double[] times = model.sol(studyRun[0]).getPVals();
      require(times.length == EXPECTED_OUTPUT_TIMES,
          "Expected " + EXPECTED_OUTPUT_TIMES + " output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(Math.abs(times[times.length - 1] - 0.03) < 1e-13, "Unexpected final output time");
      int particles = particleRows(model);
      require(particles == EXPECTED_PARTICLES,
          "Expected " + EXPECTED_PARTICLES + " particles, got " + particles);
      exportHistory(model, request);
      emit("configuration", "seed", Integer.toString(seed(request)), "step_s", stepSeconds(stepNs(request)),
          "random_number_args", "UserDefined", "seed_authority",
          PHYSICS + ".bf1.i", "brownian_seed_expression",
          RunM3C2StochasticRequest.seedParameter(), "brownian_active", "true",
          "out_of_plane", "false", "brownian_viscosity",
          RunM3C2StochasticRequest.viscosityExpression(), "brownian_temperature",
          RunM3C2StochasticRequest.temperatureExpression(), "saffman_active", "false",
          "dynamic_charge_active", "true", "field_source", "canonical_exact_connectivity_P1_sectionwise",
          "initial_state_source", "candidate_realized_source_table", "integrator", "classical_rk4",
          "integrator_order", "4", "relative_tolerance", "1e-2", "wall_accuracy_order", "1",
          "output_times", Integer.toString(times.length), "particle_rows", Integer.toString(particles),
          "time_end_s", "0.03", "directory", directory(request), "source_model", SOURCE,
          "model_saved", "false", "solve_for_assertion", "PASS",
          "solve_for_physics", studyRun[1],
          "solve_for_multiphysics", studyRun[2]);
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  private static void run(String mode, boolean validation) throws Exception {
    ModelUtil.showProgress(false);
    List<String[]> requests = requests();
    if (validation) require(requests.size() == 1, "RunnerValidation requires exactly one request");
    for (String[] request : requests) runOne(request);
    emit("run_pass", "case", RunM3C2StochasticRequest.caseSlug(),
        "requests", Integer.toString(requests.size()),
        "mode", mode,
        "time_end_s", "0.03", "output_times", Integer.toString(EXPECTED_OUTPUT_TIMES),
        "model_saved", "false");
  }

  static void runValidation() throws Exception {
    require("RunnerValidation".equals(RunM3C2StochasticRequest.mode()),
        "Validation entry requires a RunnerValidation request authority");
    run("RunnerValidation", true);
  }

  public static void main(String[] args) throws Exception {
    String mode = RunM3C2StochasticRequest.mode();
    require(!"RunnerValidation".equals(mode), "RunnerValidation must use its validation entry");
    run(mode, false);
  }
}
