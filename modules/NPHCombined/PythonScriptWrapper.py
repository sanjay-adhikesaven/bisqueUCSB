from bqapi.util import save_blob
from bqapi.comm import BQSession
from bqapi.comm import BQCommError
import os
import sys
import io
import time
from lxml import etree
import optparse
import logging
from bqapi.util import *
from bqapi.comm import BQSession
from bqapi.util import fetch_blob
from source.BQ_run_module import run_module
import nibabel as nib
import pandas as pd
from source import nph_prediction
import xml.etree.ElementTree as ET

logging.basicConfig(filename='PythonScript.log',
                    filemode='a', level=logging.DEBUG)
log = logging.getLogger('bq.modules')

# Bisque Imports

# Module Custom Imports

ROOT_DIR = './'
NIFTI_IMAGE_PATH = 'source/Scans/'
sys.path.append(os.path.join(ROOT_DIR, "source/"))
results_outdir = 'source/UNet_Outputs/'


class ScriptError(Exception):
    def __init__(self, message):
        self.message = "Script error: %s" % message

    def __str__(self):
        return self.message


class PythonScriptWrapper(object):

    def preprocess(self, bq, **kw):
        """
        Pre-process the images
        """
        log.info('Pre-Process Options: %s' % (self.options))
        """
	1. Get the resource image
	2. call hist.py with bq, log, resource_url, seeds, threshold ( bq, log, **self.options.__dict__ )
	"""
        image = bq.load(self.options.resource_url)
        log.debug('kw is: %s', str(kw))

        predictor_uniq = self.options.resource_url.split('/')[-1]
        log.info("predictor_UNIQUE: %s" % (predictor_uniq))
        predictor_url = bq.service_url('image_service', path=predictor_uniq)
        log.info("predictor_URL: %s" % (predictor_url))
        predictor_path = os.path.join(kw.get('stagingPath', 'source/Scans'),
                                      self.getstrtime() + '-' + image.name + '.nii')

        predictor_path = bq.fetchblob(predictor_url, path=predictor_path)
        log.info("predictor_path: %s" % (predictor_path))
        # predictor_path = fetchImage(bq, predictor_url, dest=predictor_path, uselocalpath=True)
        # predictor_path = bq.fetch(url=predictor_url, path=predictor_path)
        reducer_uniq = self.options.reducer_url.split('/')[-1]
        reducer_url = bq.service_url('blob_service', path=reducer_uniq)
        reducer_path = os.path.join(kw.get('stagingPath', 'source/'), 'unet_model.pt')
        reducer_path = bq.fetchblob(reducer_url, path=reducer_path)

        self.nifti_file = os.path.join(
            self.options.stagingPath, NIFTI_IMAGE_PATH, image.name)
        log.info("process image as %s" % (self.nifti_file))
        log.info("image meta: %s" % (image))
        # ip = image.pixels().format('nii.gz').fetch()
        # log.info("BRUHHHHHH IP: %s" % (type(ip)))
        # ip2 = image.pixels() #.format('nii')
        # log.info("BRUHHHHHH IP2: %s" % (type(ip2)))
        # log.info("BRUHHHHHH Value: %s" % (image.value))
        # img_nib = nib.load(ip)
        meta = image.pixels().meta().fetch()
        # meta = ET.XML(meta)
        meta = bq.factory.string2etree(meta)

        # with open(self.nifti_file, 'wb') as f:
        #    f.write(ip2.fetch())
        log.info('Executing Histogram match')
        pred = nph_prediction.main()
        log.info('Completed Histogram match')
        return pred

    def getstrtime(self):
        # format timestamp
        ts = time.gmtime()
        ts_str = time.strftime("%Y-%m-%dT%H-%M-%S", ts)
        return ts_str

    def uploadimgservice(self, bq, files):
        """
        Upload mask to image_service upon post process
        """
        mex_id = bq.mex.uri.split('/')[-1]
        log.info("BRUHHHHHH: %s" % (files))
        filename = os.path.basename(files[0])
        log.info('Up Mex: %s' % (mex_id))
        log.info('Up File: %s' % (filename))
        resource = etree.Element(
            'image', name='ModuleExecutions/nphprediction/' + filename)
        t = etree.SubElement(resource, 'tag', name="datetime", value=self.getstrtime())
        log.info('Creating upload xml data: %s ' %
                 str(etree.tostring(resource, pretty_print=True)))
        # os.path.join("ModuleExecutions","CellSegment3D", filename)
        filepath = files[0]
        # use import service to /import/transfer activating import service
        r = etree.XML(bq.postblob(filepath, xml=resource)).find('./')
        if r is None or r.get('uri') is None:
            bq.fail_mex(msg="Exception during upload results")
        else:
            log.info('Uploaded ID: %s, URL: %s' %
                     (r.get('resource_uniq'), r.get('uri')))
            bq.update_mex('Uploaded ID: %s, URL: %s' %
                          (r.get('resource_uniq'), r.get('uri')))
            self.furl = r.get('uri')
            self.fname = r.get('name')
            resource.set('value', self.furl)

        return resource

    def uploadtableservice(self, bq, files):
        """
        Upload mask to image_service upon post process
        """
        mex_id = bq.mex.uri.split('/')[-1]
        filename = os.path.basename(files)
        log.info('Up Mex: %s' % (mex_id))
        log.info('Up File: %s' % (filename))
        resource = etree.Element(
            'table', name='ModuleExecutions/nphprediction/' + filename)
        t = etree.SubElement(resource, 'tag', name="datetime", value=self.getstrtime())
        log.info('Creating upload xml data: %s ' %
                 str(etree.tostring(resource, pretty_print=True)))
        # os.path.join("ModuleExecutions","CellSegment3D", filename)
        filepath = files
        # use import service to /import/transfer activating import service
        r = etree.XML(bq.postblob(filepath, xml=resource)).find('./')
        if r is None or r.get('uri') is None:
            bq.fail_mex(msg="Exception during upload results")
        else:
            log.info('Uploaded ID: %s, URL: %s' %
                     (r.get('resource_uniq'), r.get('uri')))
            bq.update_mex('Uploaded ID: %s, URL: %s' %
                          (r.get('resource_uniq'), r.get('uri')))
            self.furl = r.get('uri')
            self.fname = r.get('name')
            resource.set('value', self.furl)

        return resource

    def run(self):
        """
        Run Python script
        """
        bq = self.bqSession
        log.info('***** self.options: %s' % (self.options))
        # table_service = bq.service ('table')
        # call scripts
        try:
            bq.update_mex('Pre-process the images')
            self.preprocess(bq)
        except (Exception, ScriptError) as e:
            log.exception("Exception during preprocess")
            bq.fail_mex(msg="Exception during pre-process: %s" % str(e))
            return
        try:
            print(os.listdir(results_outdir))
            self.outfiles = []
            for files in os.listdir(results_outdir):
                if files.endswith("1.nii.gz"):
                    self.outfiles.append(results_outdir + files)
            bq.update_mex('Uploading Mask result')
            self.resimage = self.uploadimgservice(bq, self.outfiles)
            bq.update_mex('Uploading Table result')
            self.voltable = self.uploadtableservice(
                bq, 'source/volumes_unet.csv')
            self.volconvtable = self.uploadtableservice(
                bq, 'source/volumes_conv_unet.csv')
            self.predtable = self.uploadtableservice(
                bq, 'source/predictions.csv')
        except (Exception, ScriptError) as e:
            log.exception("Exception during upload result")
            bq.fail_mex(msg="Exception during upload result: %s" % str(e))
            return

        input_path_dict = results_outdir
        log.info(input_path_dict)
        print(input_path_dict)
        outputs_dir_path = os.getcwd()
        try:
            bq.update_mex('Running module')
            log.info("input_path_dict log");
            log.info(input_path_dict)
            print("input_path_dict print");
            print(input_path_dict)
            self.output_data_path_dict = run_module(input_path_dict, outputs_dir_path)
        except (Exception, ScriptError) as e:
            log.exception("***** Exception while running module from BQ_run_module")
            bq.fail_mex(msg="Exception while running module from BQ_run_module: %s" % str(e))
            return

        try:
            bq.update_mex('Uploading results to Bisque')
            self.output_resources = self.upload_results(bq)
        except (Exception, ScriptError) as e:
            log.exception("***** Exception while uploading results to Bisque")
            bq.fail_mex(msg="Exception while uploading results to Bisque: %s" % str(e))
            return

        log.info('Completed the workflow: %s' % (self.resimage.get('value')))
        out_imgxml = """<tag name="Segmentation" type="image" value="%s">
			<template>
		          <tag name="label" value="Segmented Image" />
			</template>
		      </tag>""" % (str(self.resimage.get('value')))

        # format timestamp
        ts = time.gmtime()
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", ts)
        vols = pd.read_csv('source/volumes_unet.csv')
        volsConv = pd.read_csv('source/volumes_conv_unet.csv')
        preds = pd.read_csv('source/predictions.csv')
        # outputs = predict( bq, log, **self.options.__dict__ )
        # outtable_xml = table_service.store_array(maxMisorient, name='maxMisorientData')
        out_xml = """<tag name="Metadata">
		<tag name="Scan" type="string" value="%s"/>
		<tag name="Ventricle" type="string" value="%s"/>
		<tag name="Subcortical" type="string" value="%s"/>
		<tag name="White Matter" type="string" value="%s"/>
		<tag name="Volumes Table" type="resource" value="%s"/>
        <tag name="Volumes Converted Table" type="resource" value="%s"/>
		<tag name="Prediction" type="string" value="%s"/>
                    </tag>""" % (
        str(vols.Scan[0]), str(vols.Vent[0]), str(vols.Sub[0]), str(vols.White[0]), self.voltable.get('value'),
        self.volconvtable.get('value'), str(preds.columns[-1]))
        outputs = [out_imgxml, out_xml]
        log.debug(outputs)
        # save output back to BisQue
        for output in outputs:
            self.output_resources.append(output)

    """ 
    def setup(self):
        Pre-run initialization
        self.output_resources.append(output)
    """
    def setup(self):
        """
        Pre-run initialization
        """
        self.bqSession.update_mex('Initializing...')
        self.mex_parameter_parser(self.bqSession.mex.xmltree)
        self.output_resources = []

    def teardown(self):
        """
        Post the results to the mex xml
        """
        self.bqSession.update_mex('Returning results')
        outputTag = etree.Element('tag', name='outputs')
        for r_xml in self.output_resources:
            if isinstance(r_xml, str):
                r_xml = etree.fromstring(r_xml)
            res_type = r_xml.get('type', None) or r_xml.get(
                'resource_type', None) or r_xml.tag
            # append reference to output
            if res_type in ['table', 'image']:
                outputTag.append(r_xml)
                # etree.SubElement(outputTag, 'tag', name='output_table' if res_type=='table' else 'output_image', type=res_type, value=r_xml.get('uri',''))
            else:
                outputTag.append(r_xml)
                # etree.SubElement(outputTag, r_xml.tag, name=r_xml.get('name', '_'), type=r_xml.get('type', 'string'), value=r_xml.get('value', ''))
        log.debug('Output Mex results: %s' %
                  (etree.tostring(outputTag, pretty_print=True)))
        self.bqSession.finish_mex(tags=[outputTag])

    def mex_parameter_parser(self, mex_xml):
        """
            Parses input of the xml and add it to options attribute (unless already set)
            @param: mex_xml
        """
        # inputs are all non-"script_params" under "inputs" and all params under "script_params"
        mex_inputs = mex_xml.xpath(
            'tag[@name="inputs"]/tag[@name!="script_params"] | tag[@name="inputs"]/tag[@name="script_params"]/tag')
        if mex_inputs:
            for tag in mex_inputs:
                # skip system input values
                if tag.tag == 'tag' and tag.get('type', '') != 'system-input':
                    if not getattr(self.options, tag.get('name', ''), None):
                        log.debug('Set options with %s as %s' %
                                  (tag.get('name', ''), tag.get('value', '')))
                        setattr(self.options, tag.get(
                            'name', ''), tag.get('value', ''))
        else:
            log.debug('No Inputs Found on MEX!')

    def upload_service(self, bq, filename, data_type='image'):
        """
        Upload resource to specific service upon post process
        """
        mex_id = bq.mex.uri.split('/')[-1]

        log.info('Up Mex: %s' % (mex_id))
        log.info('Up File: %s' % (filename))
        resource = etree.Element(
            data_type, name='ModuleExecutions/' + self.module_name + '/' + filename)
        t = etree.SubElement(resource, 'tag', name="datetime", value='time')
        log.info('Creating upload xml data: %s ' %
                 str(etree.tostring(resource, pretty_print=True)))
        # os.path.join("ModuleExecutions","CellSegment3D", filename)
        filepath = filename
        # use import service to /import/transfer activating import service
        r = etree.XML(bq.postblob(filepath, xml=resource)).find('./')
        if r is None or r.get('uri') is None:
            bq.fail_mex(msg="Exception during upload results")
        else:
            log.info('Uploaded ID: %s, URL: %s' %
                     (r.get('resource_uniq'), r.get('uri')))
            bq.update_mex('Uploaded ID: %s, URL: %s' %
                          (r.get('resource_uniq'), r.get('uri')))
            self.furl = r.get('uri')
            self.fname = r.get('name')
            resource.set('value', self.furl)

        return resource

    def validate_input(self):
        """
            Check to see if a mex with token or user with password was provided.
            @return True is returned if validation credention was provided else
            False is returned
        """
        if (self.options.mexURL and self.options.token):  # run module through engine service
            return True
        # run module locally (note: to test module)
        if (self.options.user and self.options.pwd and self.options.root):
            return True
        log.debug('Insufficient options or arguments to start this module')
        return False

    def fetch_input_resources(self, bq, inputs_dir_path):  # TODO Not hardcoded resource_url
        """
        Reads input resources from xml, fetches them from Bisque, and copies them to module container for inference

        """

        log.info('***** Options: %s' % (self.options))

        input_bq_objs = []
        input_path_dict = {}  # Dictionary that contains the paths of the input resources

        inputs_tag = self.root.find("./*[@name='inputs']")
        #        print(inputs_tag)
        for input_resource in inputs_tag.findall("./*[@type='resource']"):
            # for child in node.iter():
            print(input_resource.tag, input_resource.attrib)

            input_name = input_resource.attrib['name']
            # log.info(f"***** Processing resource named: {input_name}")
            log.info("***** Processing resource named: %s" % input_name)
            resource_obj = bq.load(getattr(self.options, input_name))
            """
            bq.load returns bqapi.bqclass.BQImage object. Ex:
            resource_obj: (image:name=whale.jpeg,value=file://admin/2022-02-25/whale.jpeg,type=None,uri=http://128.111.185.163:8080/data_service/00-pkGCYS4SPCtQVcdZUUj4sX,ts=2022-02-25T17:05:13.289578,resource_uniq=00-pkGCYS4SPCtQVcdZUUj4sX)

            resource_obj: (resource:name=yolov5s.pt,type=None,uri=http://128.111.185.163:8080/data_service/00-D9e6xVPhU93JtZjZZtwkLm,ts=2022-02-26T01:08:26.198330,resource_uniq=00-D9e6xVPhU93JtZjZZtwkLm) (PythonScriptWrapper.py:137)

            resource_obj: (resource:name=test.npy,type=None,uri=http://128.111.185.163:8080/data_service/00-EC53Rcbj8do86aXpea2cgW,ts=2022-02-26T01:17:12.312780,resource_uniq=00-EC53Rcbj8do86aXpea2cgW) (PythonScriptWrapper.py:137)
            """

            input_bq_objs.append(resource_obj)
            # log.info(f"***** resource_obj: {resource_obj}")
            log.info("***** resource_obj: %s" % resource_obj)
            # log.info(f"***** resource_obj.uri: {resource_obj.uri}")
            log.info("***** resource_obj.uri: %s" % resource_obj.uri)
            # log.info(f"***** type(resource_obj): {type(resource_obj)}")
            log.info("***** type(resource_obj): %s" % type(resource_obj))

            # Append uri to dictionary of input paths
            input_path_dict[input_name] = os.path.join(inputs_dir_path, resource_obj.name)

            # Saves resource to module container at specified dest path
            # fetch_blob_output = fetch_blob(bq, resource_obj.uri, dest=input_path_dict[input_name])
            # log.info("***** fetch_blob_output: %s"  % fetch_blob_output)

            ##########################################################################################
            image = bq.load(resource_obj.uri)
            # name = image.name or next_name("blob")
            name = image.name
            log.info("***** image.resource_uniq: %s" % image.resource_uniq)
            log.info("***** image: %s" % image)
            log.info("***** image.name: %s" % image.name)
            log.info("predictor_url = bq.service_url(data_service, path = image.resource_uniq)")
            predictor_url = bq.service_url('blob_service', path=image.resource_uniq)
            log.info("predictor_URL: %s" % (predictor_url))

            # predictor_path = os.path.join(kw.get('stagingPath', 'source/Scans'), self.getstrtime()+'-'+image.name + '.nii')
            input_path_dict[input_name] = input_path_dict[input_name].replace(".nii.gz", ".nii")
            predictor_path = bq.fetchblob(predictor_url, path=input_path_dict[input_name])
            log.info("predictor_path: %s" % (predictor_path))
            # log.info(f"***** fetch_blob_output: {fetch_blob_output}")
            # log.info("***** fetch_blob_output: %s"  % fetch_blob_output)
            ##########################################################################################

        # log.info(f"***** Input path dictionary : {input_path_dict}")
        log.info("***** Input path dictionary : %s" % input_path_dict)

        return input_path_dict

    def upload_results(self, bq):
        """
        Reads output specs from xml and uploads results to Bisque using correct service
        """

        output_resources = []
        non_image_value = {}
        non_image_present = False

        # Get outputs tag and its nonimage child tag
        outputs_tag = self.root.find("./*[@name='outputs']")
        print(outputs_tag)
        nonimage_tag = outputs_tag.find("./*[@name='NonImage']")
        print(nonimage_tag.tag, nonimage_tag.attrib)

        # Upload each resource with the corresponding service
        for resource in (nonimage_tag.findall(".//*[@type]") + outputs_tag.findall("./*[@type='image']")):
            print(resource.tag, resource.attrib)
            print("NonImage type output with name %s" % resource.attrib['name'])
            resource_name = resource.attrib['name']
            resource_type = resource.attrib['type']
            resource_path = self.output_data_path_dict[resource_name]
            # log.info(f"***** Uploading output {resource_type} '{resource_name}' from {resource_path} ...")
            log.info("***** Uploading output %s '%s' from %s ..." % (resource_type, resource_name, resource_path))

            # Upload output resource to Bisque and get resource etree.Element
            output_etree_Element = self.upload_service(bq, resource_path, data_type=resource_type)
            # log.info(f"***** Uploaded output {resource_type} '{resource_name}' to {output_etree_Element.get('value')}")
            log.info("***** Uploaded output %s '%s' to %s" % (
                resource_type, resource_name, output_etree_Element.get('value')))

            # Set the value attribute of the each resource's tag to its corresponding resource uri
            resource.set('value', output_etree_Element.get('value'))

            # Append image outputs to output resources list
            if resource in outputs_tag.findall("./*[@type='image']"):
                output_resource_xml = ET.tostring(resource).decode('utf-8')
                output_resources.append(output_resource_xml)
            else:
                non_image_present = True
                non_image_value[resource_name] = output_etree_Element.get('value')

        # Append all nonimage outputs to NonImage tag and append it to output resource list
        if non_image_present:
            template_tag = nonimage_tag.find("./template")
            nonimage_tag.remove(template_tag)
            for resource in non_image_value:
                # ET.SubElement(nonimage_tag, 'tag', attrib={'name' : f"{resource}", 'type': 'resource', 'value': f"{non_image_value[resource]}"})
                ET.SubElement(nonimage_tag, 'tag', attrib={'name': "%s" % resource, 'type': 'resource',
                                                           'value': "%s" % non_image_value[resource]})

            output_resource_xml = ET.tostring(nonimage_tag).decode('utf-8')
            output_resources.append(output_resource_xml)

        # log.debug(f"***** Output Resources xml : output_resources = {output_resources}")
        log.debug("***** Output Resources xml : output_resources = %s" % output_resources)
        # SAMPLE LOG
        # ['<tag name="OutImage" type="image" value="http://128.111.185.163:8080/data_service/00-ExhzBeQiaX5F858qNjqXzM">\n               <template>\n                    <tag name="label" value="Edge Image" />\n               </template>\n          </tag>\n     ']
        return output_resources

    def main(self):
        parser = optparse.OptionParser()
        parser.add_option('--mex_url', dest="mexURL")
        parser.add_option('--module_dir', dest="modulePath")
        parser.add_option('--staging_path', dest="stagingPath")
        parser.add_option('--bisque_token', dest="token")
        parser.add_option('--user', dest="user")
        parser.add_option('--pwd', dest="pwd")
        parser.add_option('--root', dest="root")
        (options, args) = parser.parse_args()

        # Logging initializations
        fh = logging.FileHandler('scriptrun.log', mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)8s --- %(message)s ' +
                                      '(%(filename)s:%(lineno)s)', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        log.addHandler(fh)

        try:  # pull out the mex
            if not options.mexURL:
                options.mexURL = sys.argv[-2]
            if not options.token:
                options.token = sys.argv[-1]
        except IndexError:  # no argv were set
            pass
        if not options.stagingPath:
            options.stagingPath = ''

        # Options configuration
        log.debug('PARAMS : %s Options: %s' % (args, options))
        self.options = options
        if self.validate_input():
            # initalizes if user and password are provided
            if (self.options.user and self.options.pwd and self.options.root):
                self.bqSession = BQSession().init_local(
                    self.options.user, self.options.pwd, bisque_root=self.options.root)
                self.options.mexURL = self.bqSession.mex.uri
            # initalizes if mex and mex token is provided
            elif (self.options.mexURL and self.options.token):
                self.bqSession = BQSession().init_mex(self.options.mexURL, self.options.token)
            else:
                raise ScriptError(
                    'Insufficient options or arguments to start this module')

            # Setup the mex and sessions
            try:
                self.setup()
            except Exception as e:
                log.exception("Exception during setup")
                self.bqSession.fail_mex(
                    msg="Exception during setup: %s" % str(e))
                return
            # Execute the module functionality
            try:
                self.run()
            except (Exception, ScriptError) as e:
                log.exception("Exception during run")
                self.bqSession.fail_mex(
                    msg="Exception during run: %s" % str(e))
                return
            try:
                self.teardown()
            except (Exception, ScriptError) as e:
                log.exception("Exception during teardown")
                self.bqSession.fail_mex(
                    msg="Exception during teardown: %s" % str(e))
                return
            self.bqSession.close()
        log.debug('Session Close')


""" 
    Main entry point for test purposes
    
    # test with named argument and options at the start
    python PythonScriptWrapper.py \
    http://drishti.ece.ucsb.edu:8080/data_service/00-kDwj3vQq83vJA6SvVvVVh8 \
    15 0.05 \
    http://drishti.ece.ucsb.edu:8080/module_service/mex/00-XW6DsZR9puKj76Ezn9Mi79 \
    admin:00-XW6DsZR9puKj76Ezn9Mi79

    # Note: last two argument are the mex_url and token and remaining are parsed as options
"""
if __name__ == "__main__":
    PythonScriptWrapper().main()
