import com.comsol.model.DatasetFeature;
import com.comsol.model.Model;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;

/** Read-only inventory needed by the M3-C1 30 ms Case-P/Case-A runner. */
public final class InspectM3C1Theory100nm30ms {
  private static final String SOURCE = "source_copy.mph";

  private InspectM3C1Theory100nm30ms() {}

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3C1_30MS_INSPECT|").append(type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static String setting(DatasetFeature dataset, String name) {
    try {
      String[] values = dataset.getStringArray(name);
      if (values != null && values.length > 0) return Arrays.toString(values);
    } catch (Throwable ignored) {
      // Fall through to the scalar accessor used by other dataset properties.
    }
    return dataset.getString(name);
  }

  private static String setting(PhysicsFeature feature, String name) {
    try {
      String[] values = feature.getStringArray(name);
      if (values != null && values.length > 0) return Arrays.toString(values);
    } catch (Throwable ignored) {
      // Fall through to the scalar accessor used by other feature properties.
    }
    return feature.getString(name);
  }

  private static void inspectCase(
      Model model, String name, String physicsTag, String datasetTag, String backgroundStudy,
      String backgroundSolution) {
    require(has(model.component("comp1").physics().tags(), physicsTag),
        "Missing physics " + physicsTag);
    require(has(model.result().dataset().tags(), datasetTag),
        "Missing dataset " + datasetTag);
    require(has(model.study().tags(), backgroundStudy),
        "Missing background study " + backgroundStudy);
    require(has(model.sol().tags(), backgroundSolution),
        "Missing background solution " + backgroundSolution);
    require(!model.sol(backgroundSolution).isEmpty(),
        "Empty background solution " + backgroundSolution);

    Physics physics = model.component("comp1").physics(physicsTag);
    for (String featureTag : new String[] {
        "relg1", "pp1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm",
        "depf", "gf1", "bf1", "lf1", "wall1", "outin", "outpump", "axi1"
    }) {
      require(has(physics.feature().tags(), featureTag),
          "Missing feature " + physicsTag + "/" + featureTag);
    }

    DatasetFeature dataset = model.result().dataset(datasetTag);
    emit(
        "case", "name", name, "physics", physicsTag, "dataset", datasetTag,
        "dataset_solution", setting(dataset, "solution"),
        "posdof", setting(dataset, "posdof"), "pgeom", setting(dataset, "pgeom"),
        "physicsinterface", setting(dataset, "physicsinterface"),
        "charge_state", setting(physics.feature("pp1"), "Z"),
        "release_charge", setting(physics.feature("relg1"), "aux0_auxq"),
        "material_entities", Arrays.toString(physics.feature("wall1").selection().entities()),
        "material_condition", setting(physics.feature("wall1"), "WallCondition"),
        "inlet_entities", Arrays.toString(physics.feature("outin").selection().entities()),
        "inlet_condition", setting(physics.feature("outin"), "WallCondition"),
        "pump_entities", Arrays.toString(physics.feature("outpump").selection().entities()),
        "pump_condition", setting(physics.feature("outpump"), "WallCondition"),
        "axis_entities", Arrays.toString(physics.feature("axi1").selection().entities()),
        "background_study", backgroundStudy, "background_solution", backgroundSolution);
  }

  public static void main(String[] args) throws Exception {
    Model model = null;
    try {
      ModelUtil.showProgress(false);
      model = ModelUtil.loadCopy("M3C1Theory100nm30msInspect", SOURCE);
      inspectCase(model, "caseP", "fpt", "part_P_100nm", "std2", "sol2");
      inspectCase(model, "caseA", "fptas", "part_AS_100nm", "stdASf", "sol26");
      emit("pass", "source", SOURCE, "study_run", "false", "model_saved", "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
    }
  }
}
