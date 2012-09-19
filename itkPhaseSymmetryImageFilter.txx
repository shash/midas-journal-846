
#ifndef __itkPhaseSymmetryImageFilter_txx
#define __itkPhaseSymmetryImageFilter_txx
#include "itkPhaseSymmetryImageFilter.h"
#include <string>
#include <sstream>

namespace itk
{

	template <class TInputImage, class TOutputImage>
	PhaseSymmetryImageFilter<TInputImage,TOutputImage>::PhaseSymmetryImageFilter()
	{
		m_MultiplyImageFilter = MultiplyImageFilterType::New();
		m_DivideImageFilter = DivideImageFilterType::New();
		m_AddImageFilter = AddImageFilterType::New();
		m_AddImageFilter2 = AddImageFilterType::New();
		m_MaxImageFilter = MaxImageFilterType::New();
		m_AtanImageFilter = Atan2ImageFilterType::New();
		m_SSFilter = ShiftScaleImageFilterType::New();
		m_NegateFilter = ShiftScaleImageFilterType::New();
		m_NegateFilter2 = ShiftScaleImageFilterType::New();
		m_AcosImageFilter = AcosImageFilterType::New();
		m_C2RFilter = ComplexToRealFilterType::New();
		m_C2IFilter = ComplexToImaginaryFilterType::New();
		m_C2MFilter = ComplexToModulusFilterType::New();
		m_C2AFilter = ComplexToPhaseFilterType::New();
		m_AbsImageFilter = AbsImageFilterType::New();
		m_AbsImageFilter2 = AbsImageFilterType::New();
		m_MP2CFilter = MagnitudeAndPhaseToComplexFilterType::New();

		m_FFTFilter  = FFTFilterType::New();
		m_IFFTFilter = IFFTFilterType::New();

		m_NegateFilter->SetScale(-1.0);
		m_NegateFilter->SetShift(0.0);

		m_NegateFilter2->SetScale(-1.0);
		m_NegateFilter2->SetShift(0.0);

		//Create 2 initialze wavelengths
		m_Wavelengths.SetSize(2,TInputImage::ImageDimension);
		for( int i=0; i < TInputImage::ImageDimension; i++)
		{
			m_Wavelengths(0,i) = 10.0;
			m_Wavelengths(1,i) = 20.0;
	
		}

		//Set basic orientations
		m_Orientations.SetSize(TInputImage::ImageDimension,TInputImage::ImageDimension);
		for( int i=0; i < TInputImage::ImageDimension; i++)
		{
			for( int j=0; j < TInputImage::ImageDimension; j++)
			{
				if(i==j)
				{
					m_Orientations(i,j) = 1.0;
				}
				else
				{
					m_Orientations(i,j) = 0.0;
				}
				
			}
	
		}

		//Defaults
		m_AngleBandwidth=3.14159265;
		m_Sigma=0.55;
		m_T=10.0;
		m_Polarity=0;


	}

	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::Initialize( void )
	{
		typename TInputImage::SizeType  inputSize;
		typename TInputImage::IndexType inputIndex;
		typename Superclass::OutputImagePointer output = this->GetOutput();
		typename Superclass::InputImagePointer input = const_cast< TInputImage *>( this->GetInput() );

		inputIndex = input->GetLargestPossibleRegion().GetIndex();
		inputSize = input->GetLargestPossibleRegion().GetSize();
		int ndims = int(TInputImage::ImageDimension);

		m_SSFilter->SetInput(input);
		m_SSFilter->SetScale(0.0);
		m_SSFilter->SetShift(0.0);


		typename LogGaborFreqImageSourceType::Pointer LogGaborKernel =  LogGaborFreqImageSourceType::New();
		typename SteerableFiltersFreqImageSourceType::Pointer SteerableFilterKernel =  SteerableFiltersFreqImageSourceType::New();
		typename ButterworthKernelFreqImageSourceType::Pointer ButterworthFilterKernel =  ButterworthKernelFreqImageSourceType::New();

		//Initialize log gabor kernels
		LogGaborKernel->SetOrigin(this->GetInput()->GetOrigin());
		LogGaborKernel->SetSpacing(this->GetInput()->GetSpacing());
		LogGaborKernel->SetSize(inputSize);

		//Initialize directionality kernels
		SteerableFilterKernel->SetOrigin(this->GetInput()->GetOrigin());
		SteerableFilterKernel->SetSpacing(this->GetInput()->GetSpacing());
		SteerableFilterKernel->SetSize(inputSize);

		//Initialize low pass filter kernel
		ButterworthFilterKernel->SetOrigin(this->GetInput()->GetOrigin());
		ButterworthFilterKernel->SetSpacing(this->GetInput()->GetSpacing());
		ButterworthFilterKernel->SetSize(inputSize);
		ButterworthFilterKernel->SetCutoff(0.4);
		ButterworthFilterKernel->SetOrder(10.0);

		ArrayType wv;
		ArrayType sig;
		ArrayType orientation;
		double angleBandwidth;


		FloatImageStack tempStack;
		FloatImageStack lgStack;
		FloatImageStack sfStack;

		//Create log gabor kernels
		for( unsigned int w=0; w < m_Wavelengths.rows(); w++)
		{
			for(int i=0; i < ndims;i++)
			{	
				wv[i]=m_Wavelengths.get(w,i);
			}
			LogGaborKernel->SetWavelengths(wv);
			LogGaborKernel->SetSigma(m_Sigma);
			m_MultiplyImageFilter->SetInput1(LogGaborKernel->GetOutput());
			m_MultiplyImageFilter->SetInput2(ButterworthFilterKernel->GetOutput());
			m_MultiplyImageFilter->Update();
			lgStack.push_back(m_MultiplyImageFilter->GetOutput());
			lgStack[w]->DisconnectPipeline();
		}

		//Create directionality kernels
		for( unsigned int o=0; o < m_Orientations.rows(); o++)
		{
			for( int d=0; d<ndims; d++)
			{
				orientation[d] = m_Orientations.get(o,d);
				
			}
			SteerableFilterKernel->SetOrientation(orientation);
			SteerableFilterKernel->SetAngularBandwidth(m_AngleBandwidth);
 			SteerableFilterKernel->Update();
			sfStack.push_back(SteerableFilterKernel->GetOutput());
			sfStack[o]->DisconnectPipeline();
		}

		//Create filter bank by multiplying log gabor filters with directional filters
		DoubleFFTShiftImageFilterType::Pointer FFTShiftFilter = DoubleFFTShiftImageFilterType::New();

		for( unsigned int w=0; w < m_Wavelengths.rows(); w++)
		{
			tempStack.clear();
			for( int o=0; o<m_Orientations.rows(); o++)
			{
				m_MultiplyImageFilter->SetInput1(lgStack[w]);
				m_MultiplyImageFilter->SetInput2(sfStack[o]);
				FFTShiftFilter->SetInput(m_MultiplyImageFilter->GetOutput());
				FFTShiftFilter->Update();
				tempStack.push_back(FFTShiftFilter->GetOutput());
				tempStack[o]->DisconnectPipeline();
			}
			m_FilterBank.push_back(tempStack);
		}
		
	}


	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::GenerateData( void )
	{
		
		//typedef itk::ImageFileWriter< FloatImageType> WriterType;
		//WriterType::Pointer writer= WriterType::New();
		
		typename TInputImage::SizeType  inputSize;
		typename TInputImage::IndexType inputIndex;
		typename Superclass::OutputImagePointer output = this->GetOutput();
		typename Superclass::InputImagePointer input = const_cast< TInputImage *>( this->GetInput() );

		inputIndex = input->GetLargestPossibleRegion().GetIndex();
		inputSize = input->GetLargestPossibleRegion().GetSize();
		int ndims = int(TInputImage::ImageDimension);

		double initialPoint=0;
		double epsilon = 0.0001;

		ComplexImageType::Pointer finput = ComplexImageType::New();
		ComplexImageType1::Pointer bpinput = ComplexImageType1::New();

		m_FFTFilter->SetInput(input);
		m_FFTFilter->Update();
		finput = m_FFTFilter->GetOutput();
		finput->DisconnectPipeline();


		//Get the pixel count.  We need to divide the IFFT output by this because using the inverse FFT for
		//complex to complex doesn't seem to work.   So instead, we use the forward transform and divide by pixelNum
		double pxlCount = 1.0;
		for(int i=0; i < ndims; i++)
		{
			pxlCount = pxlCount*double(inputSize[i]);
		}


		typename FloatImageType::Pointer EnergyThisOrient = FloatImageType::New();
		typename FloatImageType::Pointer totalAmplitude = FloatImageType::New();
		typename FloatImageType::Pointer totalEnergy = FloatImageType::New();

		//Matlab style initalization, because these images accumulate over each loop
		//Therefore, they initially all zeros
		m_SSFilter->SetScale(0.0);
		m_SSFilter->SetShift(0.0);

		m_SSFilter->Update();
		totalAmplitude = m_SSFilter->GetOutput();
		totalAmplitude->DisconnectPipeline();

		m_SSFilter->Update();
		totalEnergy = m_SSFilter->GetOutput();
		totalEnergy->DisconnectPipeline();


		m_SSFilter->ReleaseDataFlagOn();
		m_C2MFilter->ReleaseDataFlagOn();
		m_C2AFilter->ReleaseDataFlagOn();
		m_MultiplyImageFilter->ReleaseDataFlagOn();
		m_MP2CFilter->ReleaseDataFlagOn();
		m_C2RFilter->ReleaseDataFlagOn();
		m_C2IFilter->ReleaseDataFlagOn();
		m_MaxImageFilter->ReleaseDataFlagOn();
		m_NegateFilter->ReleaseDataFlagOn();
		m_AbsImageFilter->ReleaseDataFlagOn();
		m_AbsImageFilter2->ReleaseDataFlagOn();
		m_IFFTFilter->ReleaseDataFlagOn();

		for( int o=0; o < m_Orientations.rows(); o++)
		{

			//Reset the energy value
			m_SSFilter->SetScale(0.0);
			m_SSFilter->SetShift(0.0);
			m_SSFilter->Update();
			EnergyThisOrient = m_SSFilter->GetOutput();
			EnergyThisOrient->DisconnectPipeline();

			for( int w=0; w < m_Wavelengths.rows(); w++)
			{

				

				//Multiply filters by the input image in fourier domain
				m_C2MFilter->SetInput(finput);
				m_C2AFilter->SetInput(finput);
				m_SSFilter->SetScale(1/pxlCount);
				m_SSFilter->SetShift(0.0);
				//Normalize the magnitude by the number of pixels
				m_SSFilter->SetInput(m_C2MFilter->GetOutput());
		
				m_MultiplyImageFilter->SetInput1( m_SSFilter->GetOutput() );
				m_MultiplyImageFilter->SetInput2( m_FilterBank[w][o] );
				
				m_MultiplyImageFilter->Update();////////////////
				
				m_MP2CFilter->SetInput1(m_MultiplyImageFilter->GetOutput());
				m_MP2CFilter->SetInput2(m_C2AFilter->GetOutput());
				m_MP2CFilter->Update();
ComplexImageType1::Pointer  comp = m_MP2CFilter->GetOutput();
				m_IFFTFilter->SetInput(m_MP2CFilter->GetOutput());
				m_IFFTFilter->Update();
				bpinput=m_IFFTFilter->GetOutput();
				bpinput->DisconnectPipeline();
				//Get mag, real and imag of the band passed images
				m_C2MFilter->SetInput(bpinput);
				m_C2RFilter->SetInput(bpinput);
				m_C2IFilter->SetInput(bpinput);

	
				m_AddImageFilter->SetInput1(m_C2MFilter->GetOutput());
				m_AddImageFilter->SetInput2(totalAmplitude);
				m_AddImageFilter->Update();
				totalAmplitude = m_AddImageFilter->GetOutput();
				totalAmplitude->DisconnectPipeline();

				
				//Use appropraite equation depending on polarity
				if(m_Polarity==0)
				{
					m_AbsImageFilter->SetInput(m_C2RFilter->GetOutput());
					m_AbsImageFilter2->SetInput(m_C2IFilter->GetOutput());

					m_NegateFilter->SetInput(m_AbsImageFilter2->GetOutput());

					m_AddImageFilter->SetInput1(m_AbsImageFilter->GetOutput());
					m_AddImageFilter->SetInput2(m_NegateFilter->GetOutput());

					m_AddImageFilter2->SetInput1(m_AddImageFilter->GetOutput());
					m_AddImageFilter2->SetInput2(EnergyThisOrient);
					
					m_AddImageFilter2->Update();
					EnergyThisOrient = m_AddImageFilter2->GetOutput();
					EnergyThisOrient->DisconnectPipeline();
				}
				else if(m_Polarity==1)
				{
					
					m_AbsImageFilter->SetInput(m_C2IFilter->GetOutput());
					m_NegateFilter->SetInput(m_AbsImageFilter->GetOutput());

					m_AddImageFilter->SetInput1(m_C2RFilter->GetOutput());
					m_AddImageFilter->SetInput2(m_NegateFilter->GetOutput());

					m_AddImageFilter2->SetInput1(m_AddImageFilter->GetOutput());
					m_AddImageFilter2->SetInput2(EnergyThisOrient);

					m_AddImageFilter2->Update();
					EnergyThisOrient = m_AddImageFilter2->GetOutput();
					EnergyThisOrient->DisconnectPipeline();
					
				}
				else if(m_Polarity==-1)
				{
					m_AbsImageFilter->SetInput(m_C2IFilter->GetOutput());
					m_NegateFilter->SetInput(m_C2RFilter->GetOutput());
					m_NegateFilter2->SetInput(m_AbsImageFilter->GetOutput());

					m_AddImageFilter->SetInput1(m_NegateFilter->GetOutput());
					m_AddImageFilter->SetInput2(m_NegateFilter2->GetOutput());

					m_AddImageFilter2->SetInput1(m_AddImageFilter->GetOutput());
					m_AddImageFilter2->SetInput2(EnergyThisOrient);

					m_AddImageFilter2->Update();
					EnergyThisOrient = m_AddImageFilter2->GetOutput();
					EnergyThisOrient->DisconnectPipeline();
				}
			}

			
			//Subtract the values below the noise threshold
			m_SSFilter->SetInput(EnergyThisOrient);
			m_SSFilter->SetScale(1.0);
			m_SSFilter->SetShift(-m_T);

			m_AddImageFilter->SetInput1(m_SSFilter->GetOutput());
			m_AddImageFilter->SetInput2(totalEnergy);
			m_AddImageFilter->Update();
			
			totalEnergy = m_AddImageFilter->GetOutput();
			totalEnergy->DisconnectPipeline();
		}

		
		//Set negative values to zero
		m_SSFilter->SetScale(0.0);
		m_SSFilter->SetShift(0.0);
		m_MaxImageFilter->SetInput1(totalEnergy);
		m_MaxImageFilter->SetInput2(m_SSFilter->GetOutput());

		//Divide total energy by total amplitude over all scles and orientations
		m_DivideImageFilter->SetInput1(m_MaxImageFilter->GetOutput());
		m_DivideImageFilter->SetInput2(totalAmplitude);

		m_DivideImageFilter->GraftOutput( this->GetOutput() );
		m_DivideImageFilter->Update();
		m_PhaseSymmetry = m_DivideImageFilter->GetOutput();
		this->GraftOutput( m_PhaseSymmetry );


		itkDebugMacro("GenerateOutputInformation End");

	}


	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::GenerateInputRequestedRegion()
	{
		itkDebugMacro("GenerateInputRequestedRegion Start");
		Superclass::GenerateInputRequestedRegion();

		if ( this->GetInput() )
		{
			typename TInputImage::RegionType RequestedRegion;
			typename TInputImage::SizeType  inputSize;
			typename TInputImage::IndexType inputIndex;
			typename TInputImage::SizeType  inputLargSize;
			typename TInputImage::IndexType inputLargIndex;
			typename TOutputImage::SizeType  outputSize;
			typename TOutputImage::IndexType outputIndex;

			outputIndex = this->GetOutput()->GetRequestedRegion().GetIndex();
			outputSize = this->GetOutput()->GetRequestedRegion().GetSize();
			inputLargSize = this->GetInput()->GetLargestPossibleRegion().GetSize();
			inputLargIndex = this->GetInput()->GetLargestPossibleRegion().GetIndex();

			for(unsigned int i=0; i<TInputImage::ImageDimension; i++)
			{
				inputSize[i] = outputSize[i];
				inputIndex[i] = outputIndex[i];
			}

			RequestedRegion.SetSize(inputSize);
			RequestedRegion.SetIndex(inputIndex);
			InputImagePointer input = const_cast< TInputImage *> ( this->GetInput() );
			input->SetRequestedRegion (RequestedRegion);
		}


		itkDebugMacro("GenerateInputRequestedRegion End");
	}


	/**
	* GenerateData Performs the accumulation
	*/
	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::GenerateOutputInformation( void )
	{
		typename TOutputImage::RegionType outputRegion;
		typename TInputImage::IndexType inputIndex;
		typename TInputImage::SizeType  inputSize;
		typename TOutputImage::SizeType  outputSize;
		typename TOutputImage::IndexType outputIndex;
		typename TInputImage::SpacingType inSpacing;
		typename TInputImage::PointType inOrigin;
		typename TOutputImage::SpacingType outSpacing;
		typename TOutputImage::PointType outOrigin;

		// Get pointers to the input and output
		typename Superclass::OutputImagePointer output = this->GetOutput();
		typename Superclass::InputImagePointer input = const_cast< TInputImage *>( this->GetInput() );

		//Return if input and output are both null
		if( !input || !output )
		{
			return;
		}

		inputIndex = input->GetLargestPossibleRegion().GetIndex();
		inputSize = input->GetLargestPossibleRegion().GetSize();
		inSpacing = input->GetSpacing();
		inOrigin = input->GetOrigin();

		// Set the LargestPossibleRegion of the output.

		for(unsigned int i = 0; i<InputImageDimension; i++)
		{
			outputSize[i]  = inputSize[i];
			outputIndex[i] = inputIndex[i];
			outSpacing[i] = inSpacing[i];
			outOrigin[i]  = inOrigin[i];
		}

		//Set the size of the output region
		outputRegion.SetSize(outputSize);
		//Set the index of the output region
		outputRegion.SetIndex(outputIndex);
		//Set the origin and spacing
		output->SetOrigin(outOrigin);
		output->SetSpacing(outSpacing);
		//Set the largest po
		output->SetLargestPossibleRegion(outputRegion);
	}


	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf(os,indent);

		//  os << indent << " Integral Filter Normalize By: " << m_Cutoff << std::endl;

	}



} // end namespace itk




#endif
