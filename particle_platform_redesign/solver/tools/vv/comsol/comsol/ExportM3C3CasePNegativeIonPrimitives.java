import com.comsol.model.Model;
import com.comsol.model.NumericalFeature;
import com.comsol.model.StudyFeature;
import com.comsol.model.TableFeature;
import com.comsol.model.physics.Physics;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/** Export producer-owned Case-P negative-ion primitive caches without saving the MPH. */
public final class ExportM3C3CasePNegativeIonPrimitives {
  private static final String SOURCE = "source_copy.mph";
  private static final int[] DOMAIN_3_BOUNDARIES =
      {6, 8, 28, 29, 32, 33, 34, 35, 36, 37, 38, 40, 41, 45, 46, 47};
  private static final String DENSITY =
      "withsol('sol2',ptp.xdintop_ptp3((ptp.n_wF_1m+ptp.n_wO_1m)/ptp.xdim),"
          + "setval(Vrf,100[V]))";
  private static final String FLUX_R =
      "withsol('sol2',ptp.xdintop_ptp3((ptp.n_wF_1m*(u+ptp.Vdr_wF_1m)"
          + "+ptp.n_wO_1m*(u+ptp.Vdr_wO_1m))/ptp.xdim),setval(Vrf,100[V]))";
  private static final String FLUX_Z =
      "withsol('sol2',ptp.xdintop_ptp3((ptp.n_wF_1m*(w+ptp.Vdz_wF_1m)"
          + "+ptp.n_wO_1m*(w+ptp.Vdz_wO_1m))/ptp.xdim),setval(Vrf,100[V]))";
  private static final String MASS_DENSITY =
      "withsol('sol2',ptp.xdintop_ptp3(((0.019[kg/mol]/N_A_const)*ptp.n_wF_1m"
          + "+(0.016[kg/mol]/N_A_const)*ptp.n_wO_1m)/ptp.xdim),setval(Vrf,100[V]))";

  private ExportM3C3CasePNegativeIonPrimitives() {}

  private static void emit(String record) {
    String line = "M3C3_PRIMITIVE_EXPORT|" + record;
    System.out.println(line);
    ModelUtil.serverLog(line);
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static void addDomainCache(
      Model model, String tag, String field, String source, String scale) {
    model.component("comp1").physics().create(tag, "CoefficientFormPDE", "geom1");
    Physics physics = model.component("comp1").physics(tag);
    physics.selection().set(3);
    physics.field("dimensionless").field(field);
    physics.field("dimensionless").component(new String[] {field});
    physics.feature("cfeq1").set("c", "0");
    physics.feature("cfeq1").set("a", "1");
    physics.feature("cfeq1").set("f", "(" + source + ")/(" + scale + ")");
    physics.feature("init1").set(field, "0");
  }

  private static void addBoundaryCache(
      Model model, String tag, String field, String source, String scale) {
    model.component("comp1").physics().create(tag, "CoefficientFormBoundaryPDE", "geom1");
    Physics physics = model.component("comp1").physics(tag);
    physics.selection().set(DOMAIN_3_BOUNDARIES);
    physics.field("dimensionless").field(field);
    physics.field("dimensionless").component(new String[] {field});
    physics.feature("cfeq1").set("c", "0");
    physics.feature("cfeq1").set("a", "1");
    physics.feature("cfeq1").set("f", "side(3,(" + source + "))/(" + scale + ")");
    physics.feature("init1").set(field, "0");
  }

  private static String solveCaches(Model model) {
    Set<String> oldSolutions = new HashSet<>(Arrays.asList(model.sol().tags()));
    model.study().create("m3c3std");
    model.study("m3c3std").create("stat", "Stationary");
    StudyFeature stationary = model.study("m3c3std").feature("stat");
    for (String tag : model.component("comp1").physics().tags()) {
      stationary.setEntry("activate", tag, tag.startsWith("m3c3") ? "on" : "off");
    }
    model.study("m3c3std").setGenPlots(false);
    model.study("m3c3std").createAutoSequences("all");
    String solution = null;
    for (String tag : model.sol().tags()) {
      if (!oldSolutions.contains(tag)) {
        require(solution == null, "cache study created more than one solution sequence");
        solution = tag;
      }
    }
    require(solution != null, "cache study did not create a solution sequence");
    model.sol(solution).runAll();
    model.result().dataset().create("m3c3dset", "Solution");
    model.result().dataset("m3c3dset").set("solution", solution);
    return solution;
  }

  private static void exportProvider(
      Model model, String tag, int dimension, int[] entities, String[] expressions,
      String output) throws Exception {
    String tableTag = tag + "Table";
    try {
      model.result().numerical().create(tag, "Eval");
      NumericalFeature feature = model.result().numerical(tag);
      feature.set("data", "m3c3dset");
      feature.selection().geom("geom1", dimension);
      feature.selection().set(entities);
      feature.set("expr", expressions);
      feature.set("unit", new String[] {"1/m^3", "1/(m^2*s)", "1/(m^2*s)", "kg/m^3"});
      double[][] coordinates = feature.getCoordinates();
      double[][][] values = feature.getData();
      require(coordinates.length == 2 && values.length == expressions.length,
          output + " returned an unexpected shape");
      int points = coordinates[0].length;
      for (int j = 0; j < values.length; j++) {
        require(values[j].length >= 1 && values[j][values[j].length - 1].length == points,
            output + " returned inconsistent values");
      }
      double[][] rows = new double[points][6];
      for (int i = 0; i < points; i++) {
        rows[i][0] = coordinates[0][i];
        rows[i][1] = coordinates[1][i];
        for (int j = 0; j < values.length; j++) {
          rows[i][j + 2] = values[j][values[j].length - 1][i];
          require(Double.isFinite(rows[i][j + 2]),
              output + " contains a nonfinite primitive");
        }
      }
      model.result().table().create(tableTag, "Table");
      TableFeature table = model.result().table(tableTag);
      table.setColumnHeaders(new String[] {"r_geom_cm", "z_geom_cm",
          "negative_ion_density_per_m3", "negative_ion_number_flux_r_per_m2_s",
          "negative_ion_number_flux_z_per_m2_s", "negative_ion_mass_density_kg_per_m3"});
      table.setTableData(rows);
      table.save(output);
      emit("provider|file=" + output + "|dimension=" + dimension + "|point_count=" + points);
    } finally {
      try {
        model.result().numerical().remove(tag);
      } catch (Throwable ignored) {
        // Preserve the export outcome.
      }
      try {
        model.result().table().remove(tableTag);
      } catch (Throwable ignored) {
        // Preserve the export outcome.
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Model model = null;
    try {
      ModelUtil.showProgress(false);
      model = ModelUtil.loadCopy("M3C3CasePNegativeIonPrimitiveExport", SOURCE);
      addDomainCache(model, "m3c3nd", "m3c3NegDensityD", DENSITY, "Nscale");
      addDomainCache(model, "m3c3grd", "m3c3NegFluxRD", FLUX_R, "Gscale");
      addDomainCache(model, "m3c3gzd", "m3c3NegFluxZD", FLUX_Z, "Gscale");
      addDomainCache(model, "m3c3mdd", "m3c3NegMassDensityD", MASS_DENSITY,
          "Nscale*Mscale");
      addBoundaryCache(model, "m3c3nb", "m3c3NegDensityB", DENSITY, "Nscale");
      addBoundaryCache(model, "m3c3grb", "m3c3NegFluxRB", FLUX_R, "Gscale");
      addBoundaryCache(model, "m3c3gzb", "m3c3NegFluxZB", FLUX_Z, "Gscale");
      addBoundaryCache(model, "m3c3mdb", "m3c3NegMassDensityB", MASS_DENSITY,
          "Nscale*Mscale");
      String solution = solveCaches(model);
      exportProvider(model, "m3c3DomainEval", 2, new int[] {3},
          new String[] {"Nscale*m3c3NegDensityD", "Gscale*m3c3NegFluxRD",
              "Gscale*m3c3NegFluxZD", "Nscale*Mscale*m3c3NegMassDensityD"},
          "domain_cache.csv");
      exportProvider(model, "m3c3BoundaryEval", 1, DOMAIN_3_BOUNDARIES,
          new String[] {"Nscale*m3c3NegDensityB", "Gscale*m3c3NegFluxRB",
              "Gscale*m3c3NegFluxZB", "Nscale*Mscale*m3c3NegMassDensityB"},
          "boundary_cache.csv");
      emit("pass|study=m3c3std|solution=" + solution
          + "|geometry_coordinate_unit=cm|model_save=false|coordinate_nudge=false"
          + "|missing_value_imputation=false");
    } catch (Throwable failure) {
      String message = failure.getMessage() == null ? failure.getClass().getName()
          : failure.getMessage().replace('\n', ' ').replace('\r', ' ').replace('|', '/');
      emit("fatal|exception=" + failure.getClass().getName() + "|message=" + message);
      failure.printStackTrace(System.out);
      if (failure instanceof Exception) throw (Exception) failure;
      throw new RuntimeException(failure);
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
    }
  }
}
