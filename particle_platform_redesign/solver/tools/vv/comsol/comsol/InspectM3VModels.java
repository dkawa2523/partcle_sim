import com.comsol.model.Model;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public final class InspectM3VModels {
  private static String clean(String value) {
    if (value == null) return "";
    return value.replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t");
  }

  private static void emit(String type, String... values) {
    StringBuilder out = new StringBuilder("M3V_JSON|{\"type\":\"");
    out.append(clean(type)).append("\"");
    for (int index = 0; index + 1 < values.length; index += 2) {
      out.append(",\"").append(clean(values[index])).append("\":\"")
          .append(clean(values[index + 1])).append("\"");
    }
    out.append("}");
    String line = out.toString();
    System.out.println(line);
    ModelUtil.serverLog(line);
  }

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static String entities(PhysicsFeature feature) {
    try { return Arrays.toString(feature.selection().entities()); }
    catch (Throwable ignored) { return "[]"; }
  }

  private static String setting(PhysicsFeature feature, String name) {
    try {
      String[] values = feature.getStringArray(name);
      if (values != null && values.length > 0) return Arrays.toString(values);
    } catch (Throwable ignored) {}
    try { return feature.getString(name); }
    catch (Throwable ignored) { return "<unavailable>"; }
  }

  private static void feature(Model model, String variant, String physicsTag,
                              String featureTag, String... properties) {
    Physics physics = model.component("comp1").physics(physicsTag);
    if (!has(physics.feature().tags(), featureTag)) return;
    PhysicsFeature current = physics.feature(featureTag);
    emit("physics_feature", "variant", variant, "physics_tag", physicsTag,
        "physics_label", physics.label(), "feature_tag", featureTag,
        "feature_label", current.label(), "entities", entities(current));
    for (String property : properties) {
      emit("physics_setting", "variant", variant, "physics_tag", physicsTag,
          "feature_tag", featureTag, "property", property,
          "value", setting(current, property));
    }
  }

  private static void parameter(Model model, String variant, String name) {
    try {
      emit("parameter", "variant", variant, "name", name,
          "expression", model.param().get(name));
    } catch (Throwable error) {
      emit("parameter_missing", "variant", variant, "name", name,
          "error", error.getMessage());
    }
  }

  private static void variable(Model model, String variant, String name) {
    for (String group : model.component("comp1").variable().tags()) {
      try {
        String expression = model.component("comp1").variable(group).get(name);
        if (expression != null && !expression.isEmpty()) {
          emit("variable", "variant", variant, "group", group, "name", name,
              "expression", expression);
          return;
        }
      } catch (Throwable ignored) {}
    }
    emit("variable_missing", "variant", variant, "name", name);
  }

  private static void inspect(String variant, String path) throws Exception {
    Model model = null;
    try {
      model = ModelUtil.loadCopy("M3V_" + variant, path);
      emit("model", "variant", variant, "path", path, "label", model.label(),
          "read_only", "true");
      emit("inventory", "variant", variant, "physics_tags",
          Arrays.toString(model.component("comp1").physics().tags()), "study_tags",
          Arrays.toString(model.study().tags()), "solution_tags",
          Arrays.toString(model.sol().tags()), "dataset_tags",
          Arrays.toString(model.result().dataset().tags()), "variable_groups",
          Arrays.toString(model.component("comp1").variable().tags()));

      for (String name : new String[] {"AS_Te", "AS_ne0", "AS_ni0", "AS_mi",
          "AS_mu_i", "AS_Vp", "AS_Vwall", "AS_Vwafer", "AS_Vdielectric",
          "AS_particle_dt", "AS_Z0", "particle_dt", "timestep", "u_eps",
          "Ti_floor", "sigmaR_p", "Mmix", "brownian_seed", "AS_brownian_seed",
          "iondrag_scale_P", "iondrag_scale_A", "AS_iondrag_scale", "C_lift_fm",
          "sigma_in", "rho_p", "epsr_p", "d0", "E2floor", "AS_dV_smooth",
          "AS_exp_min", "AS_exp_max", "AS_flux_speed_floor", "AS_focus_r0",
          "AS_focus_transition", "AS_n_floor", "AS_sheath_ramp", "AS_u_floor",
          "AS_float_drop"})
        parameter(model, variant, name);

      for (String name : new String[] {"AS_ugr", "AS_ugz", "AS_Tg", "AS_pabs",
          "AS_psi_raw", "AS_sheath_gate", "AS_psi", "AS_uB", "AS_ne",
          "AS_ui_mag", "AS_ni", "AS_rhoq", "AS_Er", "AS_Ez", "AS_TiV",
          "AS_Di", "AS_Gir", "AS_Giz", "AS_Gimag", "AS_uir", "AS_uiz",
          "AS_Ge0", "AS_lambdaD", "AS_phi1"})
        variable(model, variant, name);
      for (String name : new String[] {"pabs_d", "rho_g_d", "mu_g_d", "Er_P", "Ez_P",
          "ne_d", "ni_d", "nm_d", "Te_d", "mi_d", "uir_d", "uiz_d", "lambda_g_d",
          "lambdaD_d", "lambda_in_d", "TiV_d", "Ge0_d", "phi1_d"})
        variable(model, variant, name);

      for (String physicsTag : new String[] {"fpt", "fptas"}) {
        if (!has(model.component("comp1").physics().tags(), physicsTag)) continue;
        feature(model, variant, physicsTag, "auxq", "R", "StudyStep");
        feature(model, variant, physicsTag, "idf", "F", "SpecifyForce");
        feature(model, variant, physicsTag, "pp1", "ParticlePropertySpec", "dp", "rhop",
            "ChargeSpecification", "Z", "v");
        feature(model, variant, physicsTag, "relg1", "x0", "v0", "rt", "aux0_auxq",
            "SamplingFromDistribution", "InitialVelocity");
        feature(model, variant, physicsTag, "df1", "DragLaw", "Rarefaction_Effects",
            "sigmaR", "u_src", "rho", "mu", "StudyStep");
        feature(model, variant, physicsTag, "bf1", "i", "mu", "StudyStep");
        feature(model, variant, physicsTag, "ef1", "E", "SpecifyForceUsing");
        feature(model, variant, physicsTag, "liftfm", "F", "SpecifyForce");
        feature(model, variant, physicsTag, "depf", "F", "SpecifyForce");
        feature(model, variant, physicsTag, "thpf1", "ThermophoreticForceModel", "mg",
            "rho", "mu", "k", "Cp");
        feature(model, variant, physicsTag, "wall1", "WallCondition");
        feature(model, variant, physicsTag, "outin", "WallCondition");
        feature(model, variant, physicsTag, "outpump", "WallCondition");
        feature(model, variant, physicsTag, "axi1");
      }
      if (has(model.component("comp1").physics().tags(), "esass")) {
        feature(model, variant, "esass", "rhoAS", "rhoq");
        feature(model, variant, "esass", "waferAS", "V0");
        feature(model, variant, "esass", "wallAS", "V0");
        feature(model, variant, "esass", "dielectricAS", "V0");
        feature(model, variant, "esass", "dielectricOuterAS", "V0");
        feature(model, variant, "esass", "bulkAS", "V0");
      }
      for (String solution : model.sol().tags()) {
        String empty;
        try { empty = Boolean.toString(model.sol(solution).isEmpty()); }
        catch (Throwable ignored) { empty = "unknown"; }
        emit("solution", "variant", variant, "tag", solution, "empty", empty);
      }
      emit("model_pass", "variant", variant, "read_only", "true");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
    }
  }

  public static void main(String[] args) throws Exception {
    ModelUtil.showProgress(false);
    Map<String, String> models = new LinkedHashMap<>();
    models.put("relative_flow_screened_collection_orbital_v1",
        "icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_formal_iondrag_theory_consistent_10_30_100nm.mph");
    models.put("electric_field_aligned_image_sensitivity_v1",
        "icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_formal_iondrag_image_minimal_corrected_10_30_100nm.mph");
    for (Map.Entry<String, String> entry : models.entrySet()) {
      inspect(entry.getKey(), entry.getValue());
    }
    emit("audit_pass", "model_count", "2", "read_only", "true",
        "study_run", "false", "model_save", "false");
  }
}
