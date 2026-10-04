import com.comsol.model.DatasetFeature;
import com.comsol.model.Model;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.physics.PhysicsProp;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;

/** Read-only M3-C2 inventory of particle dimensionality and Brownian settings. */
public final class InspectM3C2ModelSemantics {
  private static final String SOURCE =
      "icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_"
          + "formal_iondrag_theory_consistent_10_30_100nm.mph";

  private InspectM3C2ModelSemantics() {}

  private static String clean(String value) {
    if (value == null) return "";
    return value.replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t");
  }

  private static void emit(String type, String... values) {
    StringBuilder out = new StringBuilder("M3C2_MODEL_SEMANTICS_JSON|{\"type\":\"");
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

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static String setting(PhysicsProp propertyGroup, String name) {
    try {
      String value = propertyGroup.getString(name);
      if (value != null && !value.isEmpty()) return value;
    } catch (Throwable ignored) {
      // Fall through to the array accessor.
    }
    try {
      return Arrays.toString(propertyGroup.getStringArray(name));
    } catch (Throwable error) {
      return "<unavailable:" + error.getClass().getSimpleName() + ">";
    }
  }

  private static String setting(PhysicsFeature feature, String name) {
    try {
      String value = feature.getString(name);
      if (value != null && !value.isEmpty()) return value;
    } catch (Throwable ignored) {
      // Fall through to the array accessor.
    }
    try {
      return Arrays.toString(feature.getStringArray(name));
    } catch (Throwable error) {
      return "<unavailable:" + error.getClass().getSimpleName() + ">";
    }
  }

  private static String setting(DatasetFeature dataset, String name) {
    try {
      String[] values = dataset.getStringArray(name);
      if (values != null && values.length > 0) return Arrays.toString(values);
    } catch (Throwable ignored) {
      // Fall through to the scalar accessor.
    }
    try {
      return dataset.getString(name);
    } catch (Throwable error) {
      return "<unavailable:" + error.getClass().getSimpleName() + ">";
    }
  }

  private static String entities(PhysicsFeature feature) {
    try {
      return Arrays.toString(feature.selection().entities());
    } catch (Throwable error) {
      return "<unavailable:" + error.getClass().getSimpleName() + ">";
    }
  }

  private static String required(String context, String value) {
    require(value != null && !value.isEmpty() && !value.startsWith("<unavailable:"),
        "Unavailable required setting " + context);
    return value;
  }

  private static void inspect(
      Model model, String caseName, String physicsTag, String datasetTag) {
    require(has(model.component("comp1").physics().tags(), physicsTag),
        "Missing physics " + physicsTag);
    require(has(model.result().dataset().tags(), datasetTag),
        "Missing dataset " + datasetTag);

    Physics physics = model.component("comp1").physics(physicsTag);
    for (String group : new String[] {
        "Formulation", "IncludeOutOfPlane", "RandomNumberArgs"
    }) {
      require(has(physics.prop().tags(), group),
          "Missing property group " + physicsTag + "/" + group);
    }
    require(has(physics.feature().tags(), "bf1"),
        "Missing Brownian feature " + physicsTag + "/bf1");

    String formulation = required(
        physicsTag + "/Formulation.Formulation",
        setting(physics.prop("Formulation"), "Formulation"));
    String includeOutOfPlane = required(
        physicsTag + "/IncludeOutOfPlane.IncludeOutOfPlane",
        setting(physics.prop("IncludeOutOfPlane"), "IncludeOutOfPlane"));
    String randomNumberArgs = required(
        physicsTag + "/RandomNumberArgs.RandomNumberArgs",
        setting(physics.prop("RandomNumberArgs"), "RandomNumberArgs"));
    emit(
        "physics_root", "case", caseName, "physics_tag", physicsTag,
        "physics_type", physics.getType(), "physics_label", physics.label(),
        "formulation", formulation, "include_out_of_plane", includeOutOfPlane,
        "random_number_args", randomNumberArgs);

    PhysicsFeature brownian = physics.feature("bf1");
    String brownianI = required(physicsTag + "/bf1.i", setting(brownian, "i"));
    String brownianMu = required(physicsTag + "/bf1.mu", setting(brownian, "mu"));
    String brownianTemperature = required(
        physicsTag + "/bf1.minput_temperature",
        setting(brownian, "minput_temperature"));
    emit(
        "brownian", "case", caseName, "physics_tag", physicsTag,
        "feature_tag", "bf1", "feature_type", brownian.getType(),
        "feature_label", brownian.label(), "active", Boolean.toString(brownian.isActive()),
        "selection", entities(brownian), "i", brownianI,
        "mu", brownianMu, "mu_mat", setting(brownian, "mu_mat"),
        "temperature", brownianTemperature,
        "temperature_source", setting(brownian, "minput_temperature_src"),
        "pressure", setting(brownian, "minput_pressure"),
        "pressure_source", setting(brownian, "minput_pressure_src"),
        "study_step", setting(brownian, "StudyStep"),
        "particles_to_affect", setting(brownian, "ParticlesToAffect"),
        "affected_particle_properties", setting(brownian, "AffectedParticleProperties"),
        "property_names", Arrays.toString(brownian.properties()));

    DatasetFeature dataset = model.result().dataset(datasetTag);
    String positionDofs = required(datasetTag + ".posdof", setting(dataset, "posdof"));
    emit(
        "particle_dataset", "case", caseName, "physics_tag", physicsTag,
        "dataset_tag", datasetTag, "dataset_type", dataset.getType(),
        "posdof", positionDofs, "pgeom", setting(dataset, "pgeom"),
        "physicsinterface", setting(dataset, "physicsinterface"),
        "solution", setting(dataset, "solution"));
  }

  public static void main(String[] args) throws Exception {
    Model model = null;
    try {
      ModelUtil.showProgress(false);
      model = ModelUtil.loadCopy("M3C2ModelSemanticsInspect", SOURCE);
      emit("model", "source", SOURCE, "label", model.label(), "load_copy", "true");
      inspect(model, "caseP_100nm", "fpt", "part_P_100nm");
      inspect(model, "caseA_100nm", "fptas", "part_AS_100nm");
      emit(
          "audit_pass", "source", SOURCE, "physics_count", "2", "dataset_count", "2",
          "load_copy", "true", "study_run", "false", "model_save", "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
    }
  }
}
