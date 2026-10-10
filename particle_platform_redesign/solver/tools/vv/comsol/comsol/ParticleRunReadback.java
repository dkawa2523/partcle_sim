import com.comsol.model.Model;
import com.comsol.model.ExportFeature;
import com.comsol.model.PropFeature;
import com.comsol.model.SolverFeature;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.physics.FeatureInfo;
import com.comsol.model.ParameterEntity;
import com.comsol.model.util.ModelUtil;

/** Raw API observations for an external preflight, never an equivalence verdict. */
public final class ParticleRunReadback {
  private ParticleRunReadback() {}

  private static String quote(String value) {
    if (value == null) return "null";
    StringBuilder result = new StringBuilder("\"");
    for (int index = 0; index < value.length(); index++) {
      char character = value.charAt(index);
      if (character == '"' || character == '\\') result.append('\\').append(character);
      else if (character < 32) result.append(String.format("\\u%04x", (int) character));
      else result.append(character);
    }
    return result.append('"').toString();
  }

  private static String strings(String[] values) {
    StringBuilder result = new StringBuilder("[");
    for (String value : values) {
      if (result.length() > 1) result.append(',');
      result.append(quote(value));
    }
    return result.append(']').toString();
  }

  private static String integers(int[] values) {
    StringBuilder result = new StringBuilder("[");
    for (int value : values) {
      if (result.length() > 1) result.append(',');
      result.append(value);
    }
    return result.append(']').toString();
  }

  private static String doubles(double[] values) {
    StringBuilder result = new StringBuilder("[");
    for (double value : values) {
      if (!Double.isFinite(value)) throw new IllegalStateException("Non-finite readback");
      if (result.length() > 1) result.append(',');
      result.append(value);
    }
    return result.append(']').toString();
  }

  private static String property(PropFeature feature, String name) {
    try {
      if (!feature.hasProperty(name)) throw new IllegalStateException("Property unavailable");
      String type = feature.getValueType(name);
      String value;
      if ("Boolean".equals(type)) value = Boolean.toString(feature.getBoolean(name));
      else if ("Int".equals(type)) value = Integer.toString(feature.getInt(name));
      else if ("String".equals(type)) value = quote(feature.getString(name));
      else if ("StringArray".equals(type)) value = strings(feature.getStringArray(name));
      else if ("IntArray".equals(type)) value = integers(feature.getIntArray(name));
      else if ("DoubleArray".equals(type)) value = doubles(feature.getDoubleArray(name));
      else if ("Double".equals(type)) {
        double number = feature.getDouble(name);
        if (!Double.isFinite(number)) throw new IllegalStateException("Non-finite readback");
        value = Double.toString(number);
      } else throw new IllegalStateException("Unsupported readback type " + type);
      return "{\"observation\":\"OBSERVED\",\"data_type\":" + quote(type)
          + ",\"observed_value\":" + value + "}";
    } catch (Exception failure) {
      return "{\"observation\":\"NOT_TESTED\",\"reason\":"
          + quote(failure.getClass().getName() + ": " + failure.getMessage()) + "}";
    }
  }

  private static String properties(PropFeature feature, String[] names) {
    StringBuilder result = new StringBuilder("{");
    for (String name : names) {
      if (result.length() > 1) result.append(',');
      result.append(quote(name)).append(':').append(property(feature, name));
    }
    return result.append('}').toString();
  }

  private static String property(ParameterEntity feature, String name) {
    try {
      if (!feature.hasProperty(name)) throw new IllegalStateException("Property unavailable");
      String type = feature.getValueType(name);
      String value;
      if ("Boolean".equals(type)) value = Boolean.toString(feature.getBoolean(name));
      else if ("String".equals(type)) value = quote(feature.getString(name));
      else if ("StringArray".equals(type)) value = strings(feature.getStringArray(name));
      else if ("DoubleArray".equals(type)) value = doubles(feature.getDoubleArray(name));
      else if ("Double".equals(type)) {
        double number = feature.getDouble(name);
        if (!Double.isFinite(number)) throw new IllegalStateException("Non-finite readback");
        value = Double.toString(number);
      } else throw new IllegalStateException("Unsupported readback type " + type);
      return "{\"observation\":\"OBSERVED\",\"data_type\":" + quote(type)
          + ",\"observed_value\":" + value + "}";
    } catch (Exception failure) {
      return "{\"observation\":\"NOT_TESTED\",\"reason\":"
          + quote(failure.getClass().getName() + ": " + failure.getMessage()) + "}";
    }
  }

  private static String properties(ParameterEntity feature, String[] names) {
    StringBuilder result = new StringBuilder("{");
    for (String name : names) {
      if (result.length() > 1) result.append(',');
      result.append(quote(name)).append(':').append(property(feature, name));
    }
    return result.append('}').toString();
  }

  private static String semanticGroup(String tag) {
    if ("axi1".equals(tag)) return "axis_meridional_fold";
    if ("outin".equals(tag)) return "gas_inlet_hold";
    if ("outpump".equals(tag)) return "pump_outlet_escape";
    if ("wall1".equals(tag)) return "material_boundary_unspecified";
    return "";
  }

  private static String boundaryFeatures(Physics physics) {
    StringBuilder result = new StringBuilder("{");
    for (String tag : physics.feature().tags()) {
      PhysicsFeature feature = physics.feature(tag);
      if (!feature.hasProperty("WallCondition")) continue;
      if (result.length() > 1) result.append(',');
      result.append(quote(tag)).append(":{\"active\":").append(feature.isActive())
          .append(",\"feature_type\":").append(quote(feature.getType()))
          .append(",\"semantic_group\":").append(quote(semanticGroup(tag)))
          .append(",\"properties\":")
          .append(properties(feature, new String[] {"WallCondition", "StudyStep"}));
      try {
        result.append(",\"boundary_ids\":").append(integers(feature.selection().entities(1)))
            .append(",\"selection_observation\":\"OBSERVED\"");
      } catch (Exception failure) {
        result.append(",\"boundary_ids\":[],\"selection_observation\":\"NOT_TESTED\"");
      }
      result.append('}');
    }
    return result.append('}').toString();
  }

  private static String forceFeatures(Physics physics) {
    StringBuilder result = new StringBuilder("{");
    for (String tag : physics.feature().tags()) {
      PhysicsFeature feature = physics.feature(tag);
      if (!(feature.hasProperty("F") || feature.hasProperty("R")
          || "df1".equals(tag) || "bf1".equals(tag) || "ef1".equals(tag) || "gf1".equals(tag))) continue;
      if (result.length() > 1) result.append(',');
      result.append(quote(tag)).append(":{\"active\":").append(feature.isActive())
          .append(",\"feature_type\":").append(quote(feature.getType()))
          .append(",\"properties\":")
          .append(properties(feature, new String[] {"F", "R", "E", "E_src", "StudyStep", "u_src", "u",
            "rho_mat", "rho", "mu_mat", "mu", "minput_temperature_src", "minput_temperature",
            "minput_pressure_src", "minput_pressure", "DragForceModel", "i"})).append('}');
    }
    return result.append('}').toString();
  }

  static String snapshot(Model model, String physicsTag, SolverFeature solver, String study) {
    Physics physics = model.component("comp1").physics(physicsTag);
    StringBuilder result = new StringBuilder("{\"physics_tag\":").append(quote(physicsTag))
        .append(",\"boundary_features\":").append(boundaryFeatures(physics))
        .append(",\"force_features\":").append(forceFeatures(physics))
        .append(",\"native_force_contributions\":").append(nativeForceContributions(physics))
        .append(",\"particle_properties\":")
        .append(properties(physics.feature("pp1"), new String[] {"ParticlePropertySpec", "dp", "rhop_mat", "rhop", "mp", "ChargeSpecification", "Z"}))
        .append(",\"physics_properties\":{");
    for (String name : new String[] {"Formulation", "IncludeOutOfPlane", "WallAccuracyOrder", "StoreExtra", "StoreParticleStatusData", "RandomNumberArgs"}) {
      if (result.charAt(result.length() - 1) != '{') result.append(',');
      result.append(quote(name)).append(':').append(property(physics.prop(name), name));
    }
    result.append('}');
    if (solver != null) result.append(",\"solver_properties\":")
        .append(properties(solver, new String[] {"odesolvertype", "timemethodexp", "erkorder", "rktimestep",
            "rtol", "tout", "tlist", "timestepping", "timesteppingtype", "atolglobal", "atolmethod"}));
    if (solver != null) {
      result.append(",\"numerical_parameter_observations\":{\"fixed_step_s\":")
          .append(parameterObservation(model, solver.getString("rktimestep")));
      if (physics.feature("bf1").isActive()) {
        result.append(",\"brownian_seed\":")
            .append(parameterObservation(model, physics.feature("bf1").getString("i")));
      }
      result.append('}');
    }
    if (study != null) result.append(",\"study_properties\":")
        .append(properties(model.study(study).feature("time"), new String[] {"tlist", "rtol", "notsol", "activate", "activateCoupling"}));
    return result.append('}').toString();
  }

  private static String parameterObservation(Model model, String expression) {
    try {
      double value = model.param().evaluate(expression);
      if (!Double.isFinite(value)) throw new IllegalStateException("Nonfinite evaluated parameter");
      return "{\"observation\":\"OBSERVED\",\"expression\":" + quote(expression)
          + ",\"observed_value\":" + value
          + ",\"unit\":" + quote(model.param().evaluateUnit(expression)) + "}";
    } catch (Exception failure) {
      return "{\"observation\":\"NOT_TESTED\",\"expression\":" + quote(expression)
          + ",\"reason\":" + quote(failure.getMessage()) + "}";
    }
  }

  private static String nativeForceContributions(Physics physics) {
    StringBuilder result = new StringBuilder("[");
    try {
      for (String tag : physics.feature().tags()) {
        if (!physics.feature(tag).isActive()) continue;
        for (FeatureInfo info : physics.feature(tag).featureInfo()) {
          for (String[] row : info.getInfoTable("Expression", "recursive", "all")) {
            if (row.length < 5 || !("fpt.Ftr".equals(row[1]) || "fpt.Ftz".equals(row[1])
                || "fptas.Ftr".equals(row[1]) || "fptas.Ftz".equals(row[1]))) continue;
            if (result.length() > 1) result.append(',');
            result.append("{\"feature\":").append(quote(tag)).append(",\"equation_view_row\":[");
            for (int i=0; i<row.length; ++i) {
              if (i>0) result.append(',');
              result.append(quote(row[i]));
            }
            result.append("]}");
          }
        }
      }
      return result.append(']').toString();
    } catch (Exception unavailable) {
      return "{\"observation\":\"NOT_TESTED\",\"reason\":" + quote(unavailable.getMessage()) + "}";
    }
  }

  private static String functionUnits(Model model, String functionName) {
    for (String tag : model.func().tags()) {
      try {
        String[][] names = model.func(tag).getStringMatrix("funcs");
        for (String[] row : names) if (row.length > 0 && functionName.equals(row[0]))
          return property(model.func(tag), "fununit");
      } catch (Exception unsupportedFeature) { }
    }
    return "{\"observation\":\"NOT_TESTED\",\"reason\":\"function metadata unavailable\"}";
  }

  private static String functionArguments(Model model, String functionName) {
    for (String tag : model.func().tags()) {
      try {
        String[][] names = model.func(tag).getStringMatrix("funcs");
        for (String[] row : names) if (row.length > 0 && functionName.equals(row[0]))
          return property(model.func(tag), "argunit");
      } catch (Exception unsupportedFeature) { }
    }
    return "{\"observation\":\"NOT_TESTED\",\"reason\":\"function metadata unavailable\"}";
  }

  static void write(Model model, String directory, String source, String companion) throws Exception {
    String parameters = "{}";
    try {
      parameters = "{\"diameter_m\":" + model.param().evaluate("d0")
          + ",\"model_base_unit_system\":" + quote(model.baseSystem().tag())
          + ",\"component_base_unit_system\":" + quote(model.component("comp1").baseSystem().tag())
          + ",\"variable_updates_disabled\":" + model.disableUpdates()
          + ",\"diameter_unit\":" + quote(model.param().evaluateUnit("d0"))
          + ",\"gas_molecular_mass_unit\":" + quote(model.param().evaluateUnit("m3c_mgas"))
          + ",\"gas_molecular_mass_kg\":" + model.param().evaluate("m3c_mgas")
          + ",\"delta\":" + model.param().evaluate("m3c_delta")
          + ",\"beta_expression\":" + quote(model.component("comp1").variable(CommonP1Epstein.VARIABLE_TAG).get("m3c_beta"))
          + ",\"brownian_viscosity_expression\":" + quote(model.component("comp1").variable(CommonP1Epstein.VARIABLE_TAG).get("m3c_muB"))
          + ",\"rho_function_units\":" + functionUnits(model, "m3c1_rhog")
          + ",\"rho_function_argument_units\":" + functionArguments(model, "m3c1_rhog")
          + ",\"temperature_function_units\":" + functionUnits(model, "m3c1_Tg")
          + ",\"temperature_function_argument_units\":" + functionArguments(model, "m3c1_Tg")
          + ",\"fixed_probe_r_m\":0.14,\"fixed_probe_z_m\":0.023"
          + ",\"resolved_beta_unit\":" + quote(model.param().evaluateUnit(
              model.component("comp1").variable(CommonP1Epstein.VARIABLE_TAG).get("m3c_beta")
              .replace("(r,z)", "(0.14[m],0.023[m])")))
          + ",\"resolved_viscosity_unit\":" + quote(model.param().evaluateUnit("("
              + model.component("comp1").variable(CommonP1Epstein.VARIABLE_TAG).get("m3c_beta")
              .replace("(r,z)", "(0.14[m],0.023[m])") + ")/(3*pi*d0)")) + "}";
    } catch (Exception failure) {
      parameters = "{\"observation\":\"NOT_TESTED\",\"reason\":" + quote(failure.getMessage()) + "}";
    }
    String json = "{\"schema_version\":1,\"tool_revision\":\"comsol_particle_actual_readback_v1\","
        + "\"comsol_version\":" + quote(ModelUtil.getComsolVersion()) + ",\"source\":" + source
        + ",\"companion\":" + companion + ",\"epstein_parameters\":" + parameters
        + ",\"boundary_event_export\":\"NOT_TESTED\",\"terminal_observations\":[],"
        + "\"assembly_rhs\":\"NOT_TESTED\",\"source_path\":\"source_copy.mph\"}";
    // COMSOL's default method security forbids java.nio filesystem access.
    // Emit observed JSON through the existing process log; the external
    // normalizer writes it, and the PowerShell executor owns source hashing.
    System.out.println("M3C_ACTUAL|directory=" + directory + "|json=" + json);
  }

  static void exportRetainedTerminalBoundaries(Model model, String dataset, String physics, String directory) {
    model.result().export().create("m3cTerminalBoundaries", "Data");
    ExportFeature export = model.result().export("m3cTerminalBoundaries");
    export.set("data", dataset);
    export.set("expr", new String[] {physics + ".pidx", "particlestatus", physics + ".st",
        "if(particlestatus==2||particlestatus==3,bndenv(dom),0)"});
    export.set("unit", new String[] {"1", "1", "s", "1"});
    export.set("descr", new String[] {"particle_id", "status", "event_time_s", "boundary_id"});
    export.set("filename", directory + "/terminal_boundary_raw.csv");
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    export.set("innerinput", "last");
    export.run();
  }
}
